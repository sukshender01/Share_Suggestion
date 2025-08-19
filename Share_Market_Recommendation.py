import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import datetime

# --------------------------------------------------
# Fetch NSE Index Symbols (using yfinance instead of Moneycontrol)
# --------------------------------------------------
def fetch_index_symbols(universe="NIFTY 50"):
    if universe == "NIFTY 50":
        index_symbol = "^NSEI"  # NIFTY 50
    elif universe == "NIFTY 100":
        index_symbol = "^CNX100"  # NIFTY 100
    else:
        st.error("Unsupported universe selected")
        return [], []

    # Download index components using yfinance
    index = yf.Ticker(index_symbol)
    components = index.constituents  # available in yfinance >=0.2.54

    if components is None:
        st.error("Failed to fetch index constituents.")
        return [], []

    company_names = components["Company"].tolist()
    symbols = components.index.tolist()
    return company_names, symbols

# --------------------------------------------------
# Compute Features
# --------------------------------------------------
def compute_features(df):
    df['return'] = df['Adj Close'].pct_change()
    df['rsi14'] = compute_rsi(df['Adj Close'], 14)
    df['sma20'] = df['Adj Close'].rolling(window=20).mean()
    df['sma50'] = df['Adj Close'].rolling(window=50).mean()
    df['volatility'] = df['return'].rolling(window=20).std()
    df['ret_60'] = df['Adj Close'].pct_change(60)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --------------------------------------------------
# Ranking Modes with simple explanations
# --------------------------------------------------
ranking_modes = {
    "Growth Focus": "Focus on high-return stocks. Higher risk, higher reward. Suggested for aggressive investors. Time horizon: 12–24 months.",
    "Balanced": "Mix of growth and stability. Moderate risk and steady returns. Suitable for medium-term investors. Time horizon: 6–12 months.",
    "Safe & Steady": "Focus on low volatility and stable performance. Lower risk, lower reward. Suitable for beginners. Time horizon: 12+ months.",
    "Quick Gains": "Short-term momentum opportunities. High risk, short-term reward. Suitable for small allocations. Time horizon: 1–3 months."
}

def rank_stocks(df, mode="Balanced"):
    if mode == "Growth Focus":
        df["score"] = df["ret_60"] - df["volatility"]
    elif mode == "Safe & Steady":
        df["score"] = -df["volatility"]
    elif mode == "Quick Gains":
        df["score"] = (df["sma20"] - df["sma50"]) / df["sma50"]
    else:  # Balanced
        df["score"] = (df["ret_60"] * 0.5) - (df["volatility"] * 0.5)
    return df.sort_values("score", ascending=False)

# --------------------------------------------------
# Generate Investment Suggestion
# --------------------------------------------------
def get_investment_plan(score, mode):
    if mode == "Growth Focus":
        return "₹50,000 – ₹1,00,000 for 1–2 years. High risk, potential high reward."
    elif mode == "Safe & Steady":
        return "₹20,000 – ₹50,000 for 1+ years. Low risk, stable returns."
    elif mode == "Quick Gains":
        return "₹10,000 – ₹25,000 for 1–3 months. High risk, short-term gains."
    else:
        return "₹30,000 – ₹75,000 for 6–12 months. Balanced approach."

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Stock Recommendation AI", layout="wide")

# Themes
themes = {
    "Light": {"bg": "#ffffff", "text": "#000000"},
    "Dark": {"bg": "#0e1117", "text": "#ffffff"},
    "Blue": {"bg": "#e6f0ff", "text": "#003366"},
    "Green": {"bg": "#e8f5e9", "text": "#1b5e20"},
}

theme_choice = st.sidebar.radio("Choose Theme", list(themes.keys()))
theme = themes[theme_choice]
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background-color: {theme['bg']};
        color: {theme['text']};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 AI-Based Stock Market Recommendation Tool")
st.write("Smart, simple, and beginner-friendly stock suggestions with investment guidance.")

universe = st.sidebar.selectbox("Select Stock Universe", ["NIFTY 50", "NIFTY 100"])
ranking_mode = st.sidebar.radio("Select Strategy", list(ranking_modes.keys()))
st.info(ranking_modes[ranking_mode])

company_names, symbols = fetch_index_symbols(universe)

if symbols:
    end = datetime.date.today()
    start = end - datetime.timedelta(days=365)
    stock_features = {}

    for sym in symbols[:10]:  # Limit for demo
        df = yf.download(sym + ".NS", start=start, end=end)
        if not df.empty:
            stock_features[sym] = compute_features(df)

    results = []
    for sym, df in stock_features.items():
        latest = df.iloc[-1]
        row = {
            "Symbol": sym,
            "Price": latest["Adj Close"],
            "RSI(14)": round(latest["rsi14"], 2),
            "Volatility": round(latest["volatility"], 4),
            "60D Return": round(latest["ret_60"], 4),
        }
        results.append(row)

    results_df = pd.DataFrame(results)
    ranked_df = rank_stocks(results_df, ranking_mode).head(5)

    # Add investment suggestions
    ranked_df["Suggested Investment Plan"] = ranked_df["score"].apply(lambda x: get_investment_plan(x, ranking_mode))

    st.subheader("📊 Top 5 Stock Recommendations")
    st.dataframe(ranked_df, use_container_width=True)
