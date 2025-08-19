import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ----------------------------------------
# CONFIG
# ----------------------------------------
st.set_page_config(page_title="Stock Suggestions App", layout="wide")

# Sidebar configuration
st.sidebar.header("⚙️ Settings")
universe = st.sidebar.radio("Select Universe", ["NIFTY 50", "NIFTY 500"])
ranking_mode = st.sidebar.selectbox("Ranking Mode", ["Composite", "3M Momentum", "Near 52W High"])
refresh_interval = st.sidebar.slider("Auto-refresh interval (seconds)", 30, 300, 60)

# Placeholder universe lists (replace with NSE symbols for NIFTY 50/500)
if universe == "NIFTY 50":
    stock_list = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
else:
    stock_list = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS", "LT.NS"]

# ----------------------------------------
# Fetch stock data
# ----------------------------------------
@st.cache_data(ttl=refresh_interval)
def fetch_data(symbols):
    end = datetime.today()
    start = end - timedelta(days=365)
    data = {}
    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start, end=end, progress=False)
            if not df.empty:
                df["Symbol"] = symbol
                data[symbol] = df
        except Exception as e:
            st.error(f"Error fetching {symbol}: {e}")
    return data

data_dict = fetch_data(stock_list)

# ----------------------------------------
# Prepare suggestions DataFrame
# ----------------------------------------
suggestions = []
for symbol, df in data_dict.items():
    last_price = df["Close"].iloc[-1]
    ret_60 = ((df["Close"].iloc[-1] / df["Close"].iloc[-60]) - 1) * 100 if len(df) > 60 else np.nan
    pct_to_52w_high = ((df["Close"].iloc[-1] / df["Close"].max()) - 1) * 100
    rsi14 = np.nan
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    if len(up) > 14:
        avg_gain = up.rolling(14).mean()
        avg_loss = down.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi14 = 100 - (100 / (1 + rs.iloc[-1]))

    composite = np.nanmean([ret_60, -abs(pct_to_52w_high), (50 - abs(rsi14 - 50))])

    suggestions.append({
        "Company Name": symbol.replace(".NS", ""),
        "Symbol": symbol,
        "Last": last_price,
        "ret_60": ret_60,
        "pct_to_52w_high": pct_to_52w_high,
        "rsi14": rsi14,
        "Composite Score": composite
    })

suggestions = pd.DataFrame(suggestions).dropna()

# Ranking logic
if ranking_mode == "Composite":
    suggestions = suggestions.sort_values("Composite Score", ascending=False)
elif ranking_mode == "3M Momentum":
    suggestions = suggestions.sort_values("ret_60", ascending=False)
elif ranking_mode == "Near 52W High":
    suggestions = suggestions.sort_values("pct_to_52w_high", ascending=True)

# ----------------------------------------
# Display Suggestions Table
# ----------------------------------------
st.subheader("📊 Suggested Stocks")

cols_to_show = [
    "Company Name", "Symbol", "Last",
    "Composite Score", "ret_60",
    "pct_to_52w_high", "rsi14"
]

existing_cols = [c for c in cols_to_show if c in suggestions.columns]

if not suggestions.empty and existing_cols:
    st.dataframe(suggestions[existing_cols])
else:
    st.warning("No matching columns found in suggestions.")

# ----------------------------------------
# Detailed Suggestions with Sparkline
# ----------------------------------------
st.subheader("💡 Detailed Reasons for Suggestions")

def get_reason(row, ranking_mode):
    reasons = []
    if ranking_mode == "Composite":
        reasons.append(f"Composite Score = {row.get('Composite Score', 'N/A'):.2f}, strong fundamentals + technicals.")
    if ranking_mode == "3M Momentum":
        reasons.append(f"3-month return (ret_60) = {row.get('ret_60', 'N/A'):.2f}%, strong momentum.")
    if ranking_mode == "Near 52W High":
        reasons.append(f"{row.get('pct_to_52w_high', 'N/A'):.2f}% away from 52W high — possible breakout.")
    if "rsi14" in row:
        if row["rsi14"] < 30:
            reasons.append(f"RSI {row['rsi14']:.2f} → Oversold, rebound likely.")
        elif row["rsi14"] > 70:
            reasons.append(f"RSI {row['rsi14']:.2f} → Overbought, caution.")
        else:
            reasons.append(f"RSI {row['rsi14']:.2f} → Neutral zone.")
    return " | ".join(reasons)

if not suggestions.empty:
    top_n = suggestions.head(5)  # Show top 5
    for _, row in top_n.iterrows():
        st.markdown(f"""
        ### 🏦 {row.get('Company Name', row.get('Symbol', 'Unknown'))}
        **Symbol:** {row.get('Symbol', 'N/A')}  
        **Last Price:** ₹{row.get('Last', 'N/A')}  

        **Reason:** {get_reason(row, ranking_mode)}
        """)

        # Add Sparkline (last 60 days)
        if row["Symbol"] in data_dict:
            df = data_dict[row["Symbol"]].tail(60)
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.plot(df.index, df["Close"], color="blue")
            ax.set_title("Last 60 Days Price Trend", fontsize=8)
            ax.tick_params(axis="x", labelsize=6)
            ax.tick_params(axis="y", labelsize=6)
            st.pyplot(fig)
