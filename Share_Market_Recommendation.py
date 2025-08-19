import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="📈 Share Market Recommendation", layout="wide")

# ------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------
st.sidebar.header("⚙️ Settings")
universe = st.sidebar.selectbox("Select Universe", ["NIFTY 50", "NIFTY 500"])
refresh_interval = st.sidebar.slider("Auto-refresh interval (seconds)", 30, 300, 60)
ranking_mode = st.sidebar.selectbox("Ranking Mode", ["Composite", "3M Momentum", "Near 52W High"])

# ------------------------------------------------------
# Stock Universe (example tickers — replace with full list)
# ------------------------------------------------------
if universe == "NIFTY 50":
    symbols = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "ICICIBANK.NS"]
else:
    symbols = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "ICICIBANK.NS", "SBIN.NS", "LT.NS"]

# ------------------------------------------------------
# Fetch Market Data
# ------------------------------------------------------
@st.cache_data(ttl=refresh_interval)
def get_data(symbols):
    data = {}
    for sym in symbols:
        try:
            df = yf.download(sym, period="6mo", interval="1d", progress=False)
            if not df.empty:
                df["Symbol"] = sym
                data[sym] = df
        except Exception as e:
            st.warning(f"Error fetching {sym}: {e}")
    return data

market_data = get_data(symbols)

# ------------------------------------------------------
# Compute Metrics
# ------------------------------------------------------
suggestions = []
for sym, df in market_data.items():
    try:
        last_price = df["Close"].iloc[-1]
        ret_60 = (last_price / df["Close"].iloc[-60] - 1) * 100 if len(df) > 60 else np.nan
        high_52w = df["High"].max()
        pct_to_52w_high = ((high_52w - last_price) / high_52w) * 100 if high_52w else np.nan
        # RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi14 = 100 - (100 / (1 + rs.iloc[-1])) if not loss.isna().all() else np.nan

        composite_score = (
            (ret_60 if not np.isnan(ret_60) else 0)
            - (pct_to_52w_high if not np.isnan(pct_to_52w_high) else 0)
            + ((50 - abs(rsi14 - 50)) if not np.isnan(rsi14) else 0)
        )

        suggestions.append({
            "Company Name": sym.replace(".NS", ""),
            "Symbol": sym,
            "Last": round(last_price, 2),
            "Composite Score": round(composite_score, 2),
            "ret_60": round(ret_60, 2) if not np.isnan(ret_60) else None,
            "pct_to_52w_high": round(pct_to_52w_high, 2) if not np.isnan(pct_to_52w_high) else None,
            "rsi14": round(rsi14, 2) if not np.isnan(rsi14) else None,
            "data": df  # keep full dataframe for sparkline
        })
    except Exception as e:
        st.warning(f"Error processing {sym}: {e}")

suggestions = pd.DataFrame(suggestions)

# ------------------------------------------------------
# Ranking
# ------------------------------------------------------
if ranking_mode == "Composite":
    suggestions = suggestions.sort_values("Composite Score", ascending=False)
elif ranking_mode == "3M Momentum":
    suggestions = suggestions.sort_values("ret_60", ascending=False)
elif ranking_mode == "Near 52W High":
    suggestions = suggestions.sort_values("pct_to_52w_high", ascending=True)

# ------------------------------------------------------
# Display Suggestions Table
# ------------------------------------------------------
st.subheader("📊 Suggested Stocks")

cols_to_show = ["Company Name", "Symbol", "Last", "Composite Score", "ret_60", "pct_to_52w_high", "rsi14"]
existing_cols = [c for c in cols_to_show if c in suggestions.columns]

if not suggestions.empty and existing_cols:
    st.dataframe(suggestions[existing_cols])
else:
    st.warning("No matching columns found in suggestions.")

# ------------------------------------------------------
# Detailed Suggestions with Sparkline
# ------------------------------------------------------
st.subheader("💡 Detailed Reasons for Suggestions")

def get_reason(row, ranking_mode):
    reasons = []
    if ranking_mode == "Composite":
        reasons.append(f"Composite Score = {row.get('Composite Score', 'N/A')} → Balanced technical & fundamental strength.")
    if ranking_mode == "3M Momentum":
        reasons.append(f"3-month return = {row.get('ret_60', 'N/A')}% → Strong upward momentum.")
    if ranking_mode == "Near 52W High":
        reasons.append(f"{row.get('pct_to_52w_high', 'N/A')}% away from 52W high → Possible breakout candidate.")
    if "rsi14" in row and row["rsi14"] is not None:
        if row["rsi14"] < 30:
            reasons.append(f"RSI {row['rsi14']} → Oversold, rebound possible.")
        elif row["rsi14"] > 70:
            reasons.append(f"RSI {row['rsi14']} → Overbought, risk of pullback.")
        else:
            reasons.append(f"RSI {row['rsi14']} → Neutral zone.")
    return " | ".join(reasons)

if not suggestions.empty:
    top_n = suggestions.head(5)  # Show top 5 detailed suggestions
    for _, row in top_n.iterrows():
        st.markdown(f"""
        ### 🏦 {row.get('Company Name', row.get('Symbol', 'Unknown'))}
        **Symbol:** {row.get('Symbol', 'N/A')}  
        **Last Price:** ₹{row.get('Last', 'N/A')}  

        **Reason:** {get_reason(row, ranking_mode)}
        """)
        
        # Sparkline for last 60 days
        df = row["data"]
        if len(df) > 60:
            prices = df["Close"].tail(60)
        else:
            prices = df["Close"]
        fig, ax = plt.subplots(figsize=(5,1.5))
        ax.plot(prices.index, prices.values, color="blue", linewidth=2)
        ax.fill_between(prices.index, prices.values, prices.values.min(), color="blue", alpha=0.1)
        ax.axis("off")
        st.pyplot(fig)
