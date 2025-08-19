import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
import random

# ----------------------------
# Theme Selection
# ----------------------------
theme_choice = st.sidebar.selectbox(
    "🎨 Select Theme",
    ["Light", "Dark", "Minimal", "Vibrant"]
)

if theme_choice == "Dark":
    st.markdown(
        """
        <style>
        .stApp {background-color: #111; color: #eee;}
        </style>
        """, unsafe_allow_html=True
    )
elif theme_choice == "Minimal":
    st.markdown(
        """
        <style>
        .stApp {background-color: #f7f7f7; color: #333;}
        </style>
        """, unsafe_allow_html=True
    )
elif theme_choice == "Vibrant":
    st.markdown(
        """
        <style>
        .stApp {background-color: #fff0f5; color: #222;}
        </style>
        """, unsafe_allow_html=True
    )

# ----------------------------
# Ranking Explanation
# ----------------------------
ranking_explanations = {
    "Composite Score": "This ranks stocks based on a mix of growth, stability, and momentum. Good for medium investment with balanced risk.",
    "3M Momentum": "This checks performance over the past 3 months. Higher momentum = higher chance of quick profits, but with more risk. Best for short to medium term investment.",
    "Near 52W High": "Stocks close to their 52-week high usually show strength. Safer bets, good for medium to long-term holding."
}

# ----------------------------
# Fetch Data
# ----------------------------
@st.cache_data
def load_data(symbols):
    data = yf.download(symbols, period="6mo", interval="1d", progress=False, group_by='ticker')
    return data

symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]  # sample universe
data = load_data(symbols)

# ----------------------------
# Feature Engineering
# ----------------------------
def compute_features(df):
    df['ret_60'] = df['Adj Close'].pct_change(60)
    df['52w_high'] = df['High'].rolling(252, min_periods=1).max()
    df['pct_to_52w_high'] = (df['Adj Close'] / df['52w_high'] - 1) * 100
    df['rsi14'] = compute_rsi(df['Adj Close'], 14)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

stock_features = {}
for sym in symbols:
    df = data[sym].copy()
    stock_features[sym] = compute_features(df)

# ----------------------------
# Ranking Function
# ----------------------------
def rank_stocks(ranking_mode="Composite Score"):
    scores = []
    for sym, df in stock_features.items():
        last_row = df.iloc[-1]
        try:
            if ranking_mode == "Composite Score":
                score = (
                    0.4 * last_row["ret_60"] +
                    0.3 * (-last_row["pct_to_52w_high"]) +
                    0.3 * (50 - abs(50 - last_row["rsi14"]))
                )
            elif ranking_mode == "3M Momentum":
                score = last_row["ret_60"]
            elif ranking_mode == "Near 52W High":
                score = -last_row["pct_to_52w_high"]
            else:
                score = 0
        except Exception:
            score = 0
        scores.append((sym, score, last_row))
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return ranked

# ----------------------------
# Reasoning Function
# ----------------------------
def get_reason(row, ranking_mode):
    reasons = []
    if "rsi14" in row and pd.notna(row["rsi14"]):
        if row["rsi14"] < 30:
            reasons.append("Stock looks oversold → could bounce back.")
        elif row["rsi14"] > 70:
            reasons.append("Stock is overbought → might cool off.")
    if "ret_60" in row and pd.notna(row["ret_60"]):
        if row["ret_60"] > 0:
            reasons.append("It has shown gains in the last 3 months.")
        else:
            reasons.append("It has lost in the last 3 months.")
    if "pct_to_52w_high" in row and pd.notna(row["pct_to_52w_high"]):
        if row["pct_to_52w_high"] > -5:
            reasons.append("Close to its yearly high → strong momentum.")
        else:
            reasons.append("Far from yearly high → may recover if trend reverses.")
    return " ".join(reasons)

# ----------------------------
# Investment Suggestion
# ----------------------------
def get_investment_plan(row, ranking_mode):
    amount_options = ["₹10,000", "₹25,000", "₹50,000+"]
    duration = {
        "Composite Score": "6–12 months (medium term)",
        "3M Momentum": "1–3 months (short term, quick gains)",
        "Near 52W High": "1–2 years (long term hold)"
    }
    risk = {
        "Composite Score": "Moderate risk with balanced return potential.",
        "3M Momentum": "High risk, but can deliver faster returns.",
        "Near 52W High": "Lower risk, steady returns expected."
    }
    chosen_amount = random.choice(amount_options)
    return f"💰 Suggested Investment: {chosen_amount}\n⏳ Suggested Duration: {duration[ranking_mode]}\n⚠️ Risk Level: {risk[ranking_mode]}"

# ----------------------------
# Sparkline Plot
# ----------------------------
def plot_sparkline(prices):
    fig, ax = plt.subplots(figsize=(4, 1))
    ax.plot(prices, color="green")
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return buf

# ----------------------------
# UI Layout
# ----------------------------
st.title("📈 Smart Stock Suggestions (India)")

ranking_mode = st.selectbox("Ranking Mode", ["Composite Score", "3M Momentum", "Near 52W High"])
st.markdown(f"ℹ️ **{ranking_explanations[ranking_mode]}**")

ranked = rank_stocks(ranking_mode)

st.subheader("📊 Suggested Stocks")

cols = st.columns(2)
for i, (sym, score, row) in enumerate(ranked[:6]):
    col = cols[i % 2]
    with col:
        st.markdown(f"### {sym}")
        st.image(plot_sparkline(stock_features[sym]['Adj Close'].tail(60)))
        st.markdown(f"**Reason:** {get_reason(row, ranking_mode)}")
        st.markdown(get_investment_plan(row, ranking_mode))
