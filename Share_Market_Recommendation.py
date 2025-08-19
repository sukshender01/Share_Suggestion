import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# --------------------------------------------
# CONFIG
# --------------------------------------------
st.set_page_config(page_title="Stock Suggestion App", layout="wide")

# Themes
THEMES = {
    "Light": {"bg": "#ffffff", "text": "#000000", "card": "#f9f9f9"},
    "Dark": {"bg": "#0e1117", "text": "#fafafa", "card": "#1e222a"},
    "Blue": {"bg": "#e8f1fc", "text": "#0a1f44", "card": "#d0e2f5"},
    "Green": {"bg": "#f1fcf2", "text": "#093a0a", "card": "#d4f5d6"},
}

# Sidebar
st.sidebar.title("⚙️ Settings")
universe = st.sidebar.selectbox("Universe", ["NIFTY 50", "NIFTY 500"])
refresh_mins = st.sidebar.slider("Auto-refresh interval (minutes)", 1, 30, 5)
ranking_mode = st.sidebar.selectbox(
    "Ranking Mode",
    ["Composite Score", "3M Momentum", "Near 52W High"],
    help="Select how stocks should be ranked",
)
theme_choice = st.sidebar.selectbox("Theme", list(THEMES.keys()))

# Apply theme
theme = THEMES[theme_choice]
st.markdown(
    f"""
    <style>
    body {{
        background-color: {theme['bg']};
        color: {theme['text']};
    }}
    .stApp {{
        background-color: {theme['bg']};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Ranking explanations
ranking_explanations = {
    "Composite Score": "📊 A balanced score combining return, momentum, and risk. Good for moderate investors who want steady growth over 6–12 months.",
    "3M Momentum": "🚀 Focuses on stocks that have been rising in the last 3 months. Good for short-term traders (3–6 months) but risk is higher.",
    "Near 52W High": "📈 Stocks close to their yearly peak, often indicating strength. Medium-term investors (6–12 months) can consider, but risk of pullback exists.",
}

# --------------------------------------------
# FUNCTIONS
# --------------------------------------------
def fetch_index_symbols(universe):
    if universe == "NIFTY 50":
        tickers = pd.read_html("https://www.moneycontrol.com/markets/indian-indices/top-nse-50-companies-list/9/7")[0]
        return tickers["Company Name"].tolist(), tickers["Symbol"].tolist()
    else:
        tickers = pd.read_html("https://www.moneycontrol.com/markets/indian-indices/top-nse-500-companies-list/9/23")[0]
        return tickers["Company Name"].tolist(), tickers["Symbol"].tolist()

def compute_features(df):
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    features = {}
    df["ret_60"] = df["Adj Close"].pct_change(60)
    df["ret_20"] = df["Adj Close"].pct_change(20)

    # RSI (14)
    delta = df["Adj Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    # % to 52W high
    high_52w = df["Adj Close"].rolling(252).max()
    df["pct_to_52w_high"] = (high_52w - df["Adj Close"]) / high_52w * 100

    features["ret_60"] = df["ret_60"].iloc[-1]
    features["ret_20"] = df["ret_20"].iloc[-1]
    features["rsi14"] = df["rsi14"].iloc[-1]
    features["pct_to_52w_high"] = df["pct_to_52w_high"].iloc[-1]
    features["last_price"] = df["Adj Close"].iloc[-1]

    return features, df

def get_reason(row, mode):
    if mode == "Composite Score":
        if row["rsi14"] < 30:
            return "Stock looks oversold and might rebound. Moderate investment for 6–12 months."
        elif row["pct_to_52w_high"] < 10:
            return "Stock is near yearly high, showing strength. Consider investing moderately for 6–9 months."
        else:
            return "Balanced performer with growth potential. Good for steady investment."
    elif mode == "3M Momentum":
        if row["ret_60"] > 0.1:
            return "Strong upward momentum in last 3 months. Suitable for short-term (3–6 months) but higher risk."
        else:
            return "Moderate momentum, potential upside with caution."
    elif mode == "Near 52W High":
        if row["pct_to_52w_high"] < 5:
            return "Stock is very close to 52W high, trend is strong. Good for 6–12 month investment."
        else:
            return "Approaching highs, but slightly risky. Small allocation advised."
    return "General suggestion based on ranking."

def investment_advice(row):
    if row["rsi14"] < 30:
        return "💡 Suggestion: Invest ₹50,000 for 6–12 months. Low downside, potential recovery."
    elif row["ret_60"] > 0.1:
        return "💡 Suggestion: Invest ₹30,000 for 3–6 months. Momentum play, but higher risk."
    elif row["pct_to_52w_high"] < 10:
        return "💡 Suggestion: Invest ₹40,000 for 6–9 months. Strong trend but watch for corrections."
    else:
        return "💡 Suggestion: Invest ₹25,000 for 6 months. Moderate risk and return."

def plot_sparkline(df):
    fig, ax = plt.subplots(figsize=(3, 1))
    ax.plot(df.index, df["Adj Close"], color="blue")
    ax.axis("off")
    return fig

# --------------------------------------------
# MAIN
# --------------------------------------------
st.title("📈 Stock Market Suggestion App")
st.caption("Real-time suggestions for Indian stocks with easy explanations")

st.markdown(f"### ℹ️ Ranking Method Explanation\n{ranking_explanations[ranking_mode]}")

company_names, symbols = fetch_index_symbols(universe)

stock_features = {}
dfs = {}
for sym in symbols[:20]:  # limit to first 20 for speed
    try:
        df = yf.download(sym + ".NS", period="6mo", interval="1d", progress=False)
        if not df.empty:
            features, full_df = compute_features(df)
            stock_features[sym] = features
            dfs[sym] = full_df
    except Exception:
        continue

df_features = pd.DataFrame(stock_features).T

if df_features.empty:
    st.error("No data fetched. Please try again.")
else:
    if ranking_mode == "Composite Score":
        df_features["score"] = (
            df_features["ret_60"].rank(ascending=False)
            + (-df_features["pct_to_52w_high"]).rank(ascending=True)
            + df_features["rsi14"].rank(ascending=True)
        )
    elif ranking_mode == "3M Momentum":
        df_features["score"] = df_features["ret_60"]
    elif ranking_mode == "Near 52W High":
        df_features["score"] = -df_features["pct_to_52w_high"]

    suggestions = df_features.sort_values("score", ascending=False).head(5)

    st.subheader("📊 Top Stock Suggestions")
    cols = st.columns(2)
    for i, (sym, row) in enumerate(suggestions.iterrows()):
        with cols[i % 2]:
            st.markdown(
                f"""
                ### {sym}
                **Last Price:** ₹{row['last_price']:.2f}  
                **Reason:** {get_reason(row, ranking_mode)}  
                {investment_advice(row)}
                """,
                unsafe_allow_html=True,
            )
            st.pyplot(plot_sparkline(dfs[sym]), use_container_width=True)
