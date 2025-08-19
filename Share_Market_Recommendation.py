import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------
# Configurations
# -----------------------------
st.set_page_config(page_title="Smart Share Market Recommendations", layout="wide")

# -----------------------------
# Themes for UI
# -----------------------------
themes = {
    "Light": {"bgcolor": "#FFFFFF", "textcolor": "#000000"},
    "Dark": {"bgcolor": "#1E1E1E", "textcolor": "#FFFFFF"},
    "Blue": {"bgcolor": "#E6F0FF", "textcolor": "#003366"},
    "Green": {"bgcolor": "#E9F7EF", "textcolor": "#145A32"},
}
theme_choice = st.sidebar.selectbox("🎨 Choose Theme", list(themes.keys()))
st.markdown(
    f"""
    <style>
        .reportview-container {{
            background-color: {themes[theme_choice]["bgcolor"]};
            color: {themes[theme_choice]["textcolor"]};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Fetch index symbols
# -----------------------------
def fetch_index_symbols(universe="NIFTY50"):
    try:
        index = yf.Ticker("^NSEI") if universe == "NIFTY50" else yf.Ticker("^BSESN")
        if hasattr(index, "constituents"):
            comp = index.constituents
            names = list(comp.keys())
            syms = list(comp.values())
            return names, syms
    except Exception:
        pass

    # fallback: static list of NIFTY50 top companies
    nifty50 = {
        "Reliance Industries": "RELIANCE.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "Infosys": "INFY.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "TCS": "TCS.NS",
        "Kotak Mahindra Bank": "KOTAKBANK.NS",
        "Axis Bank": "AXISBANK.NS",
        "ITC": "ITC.NS",
        "Bharti Airtel": "BHARTIARTL.NS",
        "Hindustan Unilever": "HINDUNILVR.NS"
    }
    return list(nifty50.keys()), list(nifty50.values())

# -----------------------------
# Compute features
# -----------------------------
def compute_features(df):
    df["ret_20"] = df["Adj Close"].pct_change(20)
    df["ret_60"] = df["Adj Close"].pct_change(60)
    df["volatility"] = df["Adj Close"].pct_change().rolling(20).std()
    return df

# -----------------------------
# Ranking method
# -----------------------------
def rank_stocks(features):
    scores = {}
    for sym, df in features.items():
        if df.empty:
            continue
        latest = df.iloc[-1]
        score = (
            (latest["ret_20"] * 0.4) +
            (latest["ret_60"] * 0.4) -
            (latest["volatility"] * 0.2)
        )
        scores[sym] = score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# -----------------------------
# Investment suggestion generator
# -----------------------------
def generate_suggestions(ranked, names_map):
    suggestions = []
    for i, (sym, score) in enumerate(ranked[:5], start=1):
        name = names_map.get(sym, sym)
        if i == 1:
            risk, amt, dur = "Low", "₹1,00,000", "3 years"
        elif i == 2:
            risk, amt, dur = "Moderate", "₹75,000", "2-3 years"
        elif i == 3:
            risk, amt, dur = "Moderate", "₹50,000", "2 years"
        elif i == 4:
            risk, amt, dur = "High", "₹25,000", "1-2 years"
        else:
            risk, amt, dur = "High", "₹10,000", "1 year"

        suggestions.append({
            "Rank": i,
            "Company": name,
            "Symbol": sym,
            "Score": round(score, 3),
            "Suggested Investment": amt,
            "Time Duration": dur,
            "Risk": risk,
            "Description": f"{name} is ranked #{i}. Risk: {risk}. Suggested to invest {amt} for {dur} with potential returns depending on market conditions."
        })
    return suggestions

# -----------------------------
# Main
# -----------------------------
st.title("📈 Smart Share Market Recommendation System")

universe = st.sidebar.selectbox("Select Universe", ["NIFTY50", "SENSEX"])
company_names, symbols = fetch_index_symbols(universe)
name_map = dict(zip(symbols, company_names))

data = {}
features = {}
for sym in symbols:
    try:
        df = yf.download(sym, period="1y", interval="1d", progress=False)
        if not df.empty:
            df = compute_features(df)
            data[sym] = df
            features[sym] = df
    except Exception as e:
        st.warning(f"Could not fetch {sym}: {e}")

ranked = rank_stocks(features)
if ranked:
    st.subheader("Top Investment Suggestions")
    suggestions = generate_suggestions(ranked, name_map)
    df_suggestions = pd.DataFrame(suggestions)
    st.dataframe(df_suggestions, use_container_width=True)

    # Chart for top stock
    top_sym = ranked[0][0]
    st.subheader(f"📊 Price Trend - {name_map.get(top_sym, top_sym)}")
    plt.figure(figsize=(10, 5))
    plt.plot(data[top_sym].index, data[top_sym]["Adj Close"])
    plt.title(f"{top_sym} Price Trend (1 Year)")
    st.pyplot(plt)
else:
    st.warning("No data available for recommendations.")
