import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# --------------------------
# THEME OPTIONS
# --------------------------
themes = {
    "Light": {"bg": "#FFFFFF", "text": "#000000", "accent": "#1E88E5"},
    "Dark": {"bg": "#0E1117", "text": "#FAFAFA", "accent": "#BB86FC"},
    "Blue": {"bg": "#E3F2FD", "text": "#0D47A1", "accent": "#1976D2"},
    "Green": {"bg": "#E8F5E9", "text": "#1B5E20", "accent": "#43A047"}
}

# --------------------------
# CONFIGURATION
# --------------------------
st.set_page_config(page_title="Indian Stock Suggestions", layout="wide")

theme_choice = st.sidebar.radio("🎨 Choose Theme", list(themes.keys()))
theme = themes[theme_choice]
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background-color: {theme['bg']};
        color: {theme['text']};
    }}
    .stMarkdown, .stDataFrame, .stRadio, .stSelectbox, .stText {{
        color: {theme['text']} !important;
    }}
    .stButton>button {{
        background-color: {theme['accent']} !important;
        color: white !important;
        border-radius: 8px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Smart Stock Suggestions (India)")

# --------------------------
# Ranking Methods Explained
# --------------------------
ranking_descriptions = {
    "Composite Score": "📊 Balanced approach. Combines momentum, distance from 52-week high, and RSI. Good for medium-risk investors. Suggested holding: 6–12 months.",
    "3M Momentum": "🚀 Focuses on stocks rising steadily in the last 3 months. High potential but medium-to-high risk. Suggested holding: 3–6 months.",
    "Near 52W High": "🔔 Stocks close to their yearly high. Often shows strength, but may have limited upside. Suggested holding: 2–4 months."
}

universe_choice = st.sidebar.radio("📌 Universe", ["NIFTY 50", "NIFTY 500"])
ranking_mode = st.sidebar.radio("📊 Ranking Mode", list(ranking_descriptions.keys()))
st.info(ranking_descriptions[ranking_mode])

refresh_interval = st.sidebar.slider("⏱ Auto-refresh (minutes)", 1, 30, 5)

# --------------------------
# STOCK UNIVERSE
# --------------------------
nifty50 = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS"]
nifty500 = nifty50 + ["SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS"]

symbols = nifty50 if universe_choice == "NIFTY 50" else nifty500

# --------------------------
# FEATURES CALCULATION
# --------------------------
def compute_features(df):
    if "Adj Close" in df.columns:
        close = df["Adj Close"]
    else:
        close = df["Close"]

    df["ret_60"] = close.pct_change(60)
    df["rolling_max"] = close.rolling(252, min_periods=1).max()
    df["pct_to_52w_high"] = (close / df["rolling_max"]) - 1
    df["returns"] = close.pct_change()
    df["up"] = np.where(df["returns"] > 0, df["returns"], 0)
    df["down"] = np.where(df["returns"] < 0, -df["returns"], 0)
    roll_up = df["up"].rolling(14, min_periods=1).mean()
    roll_down = df["down"].rolling(14, min_periods=1).mean()
    rs = roll_up / (roll_down + 1e-6)
    df["rsi14"] = 100 - (100 / (1 + rs))
    return df

def get_reason(row, ranking_mode):
    if ranking_mode == "Composite Score":
        return f"This stock balances momentum, strength, and RSI. Suitable for medium-risk investors. Investment: ₹50,000+ for 6–12 months."
    elif ranking_mode == "3M Momentum":
        return f"This stock has shown steady 3-month growth. High return potential but risky. Investment: ₹30,000+ for 3–6 months."
    elif ranking_mode == "Near 52W High":
        return f"This stock is trading near its yearly high. Can be safer, but limited upside. Investment: ₹20,000+ for 2–4 months."
    return "General investment suggestion."

# --------------------------
# FETCH DATA
# --------------------------
stock_features = {}
for sym in symbols:
    try:
        df = yf.download(sym, period="1y", interval="1d", progress=False)
        if df.empty:
            continue
        df = compute_features(df)
        stock_features[sym] = df
    except Exception as e:
        st.warning(f"⚠️ Could not fetch {sym}: {e}")

# --------------------------
# SCORE & RANK
# --------------------------
rows = []
for sym, df in stock_features.items():
    latest = df.iloc[-1]
    rows.append({
        "Symbol": sym,
        "Last": latest["Close"],
        "ret_60": latest["ret_60"],
        "pct_to_52w_high": latest["pct_to_52w_high"],
        "rsi14": latest["rsi14"]
    })

df_feat = pd.DataFrame(rows).dropna()

if not df_feat.empty:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_feat[["ret_60", "pct_to_52w_high", "rsi14"]])
    df_feat["Composite Score"] = scaled.mean(axis=1)

    if ranking_mode == "Composite Score":
        suggestions = df_feat.sort_values("Composite Score", ascending=False).head(5)
    elif ranking_mode == "3M Momentum":
        suggestions = df_feat.sort_values("ret_60", ascending=False).head(5)
    else:
        suggestions = df_feat.sort_values("pct_to_52w_high", ascending=False).head(5)

    st.subheader("📌 Top Stock Suggestions")
    st.dataframe(
        suggestions[["Symbol", "Last", "Composite Score", "ret_60", "pct_to_52w_high", "rsi14"]],
        use_container_width=True
    )

    # --------------------------
    # DETAILED GRID SUGGESTIONS
    # --------------------------
    st.subheader("📑 Detailed Recommendations")
    cols = st.columns(2)

    for i, (_, row) in enumerate(suggestions.iterrows()):
        with cols[i % 2]:
            st.markdown(f"### {row['Symbol']}")
            st.write(f"💰 Current Price: ₹{row['Last']:.2f}")
            st.write(f"📊 {get_reason(row, ranking_mode)}")

            # Sparkline
            df_hist = stock_features[row["Symbol"]].tail(60)
            fig, ax = plt.subplots(figsize=(4, 1.5))
            ax.plot(df_hist["Close"], color=theme["accent"])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Last 60 days trend", fontsize=8)
            st.pyplot(fig, use_container_width=True)
else:
    st.warning("⚠️ No stock data available right now.")
