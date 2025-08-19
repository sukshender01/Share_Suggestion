import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# ------------------------------------------------------
# THEME SELECTION
# ------------------------------------------------------
themes = {
    "Light": {"bg": "#ffffff", "text": "#000000"},
    "Dark": {"bg": "#1e1e1e", "text": "#ffffff"},
    "Blue": {"bg": "#e6f0ff", "text": "#003366"},
    "Green": {"bg": "#e9f7ef", "text": "#145a32"},
}

theme_choice = st.sidebar.selectbox("Choose Theme", list(themes.keys()))
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background-color: {themes[theme_choice]["bg"]};
        color: {themes[theme_choice]["text"]};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------
# RANKING MODE EXPLANATION
# ------------------------------------------------------
ranking_explanations = {
    "Growth": "Focuses on fast-growing stocks. Higher potential returns, but also higher risk. Suitable for short to medium-term investments (6–18 months).",
    "Value": "Focuses on undervalued stocks that may rise steadily. Lower risk than growth but takes time to show returns. Suitable for medium to long-term investments (1–3 years).",
    "Momentum": "Picks stocks already moving upward. Good for quick gains, but requires close monitoring. Suitable for short-term investments (1–6 months).",
    "Balanced": "Mix of growth, value, and momentum. Moderate risk and returns. Suitable for all types of investors with 1–2 year horizon."
}

# ------------------------------------------------------
# FETCH STOCK DATA
# ------------------------------------------------------
def fetch_stock_data(symbols, period="6mo"):
    data = {}
    for sym in symbols:
        try:
            df = yf.download(sym, period=period, progress=False)
            if not df.empty:
                data[sym] = df
        except Exception:
            pass
    return data

# ------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------
def compute_features(df):
    # Use 'Adj Close' if available, otherwise use 'Close'
    if "Adj Close" in df.columns:
        price_col = "Adj Close"
    else:
        price_col = "Close"

    df['ret_60'] = df[price_col].pct_change(60)
    df['ret_20'] = df[price_col].pct_change(20)
    df['52w_high'] = df[price_col].rolling(252).max()
    df['pct_to_52w_high'] = (df['52w_high'] - df[price_col]) / df['52w_high'] * 100
    df['ma_14'] = df[price_col].rolling(14).mean()

    # RSI Calculation
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi14'] = 100 - (100 / (1 + rs))

    return df

# ------------------------------------------------------
# SCORE CALCULATION
# ------------------------------------------------------
def calculate_score(row, mode):
    score = 0
    if mode == "Growth":
        score += row.get("ret_60", 0) * 100
    elif mode == "Value":
        score += (row.get("pct_to_52w_high", 0)) * 0.5
    elif mode == "Momentum":
        score += row.get("ret_20", 0) * 100
    elif mode == "Balanced":
        score += (row.get("ret_20", 0) * 50) + (row.get("pct_to_52w_high", 0) * 0.5)
    return score

# ------------------------------------------------------
# REASONS IN SIMPLE TERMS
# ------------------------------------------------------
def get_reason(row, mode):
    if mode == "Growth":
        if row.get("ret_60", 0) > 0.1:
            return "This stock grew strongly in the last 3 months, meaning your money could grow faster but comes with higher risk."
        else:
            return "Stock growth is moderate; safer but slower returns."
    elif mode == "Value":
        return "The stock is trading below its yearly high, meaning it may be undervalued. A patient investor may see good gains in 1–3 years."
    elif mode == "Momentum":
        return "The stock has been rising quickly recently, so you might earn quick gains in the next few months. Risk is higher if the trend breaks."
    elif mode == "Balanced":
        return "This stock balances growth and stability, meaning moderate returns with medium risk, suitable for steady investors."
    return "General investment opportunity."

# ------------------------------------------------------
# SPARKLINE (TREND CHART)
# ------------------------------------------------------
def create_sparkline(df):
    if "Adj Close" in df.columns:
        price_col = "Adj Close"
    else:
        price_col = "Close"

    fig, ax = plt.subplots(figsize=(4, 1))
    ax.plot(df.index[-60:], df[price_col].tail(60), color="blue")
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"

# ------------------------------------------------------
# INVESTMENT SUGGESTION (AMOUNT + DURATION)
# ------------------------------------------------------
def investment_plan(row, mode):
    if mode == "Growth":
        return "💰 Suggested Investment: ₹50,000 – ₹1,00,000\n⏳ Duration: 6–18 months"
    elif mode == "Value":
        return "💰 Suggested Investment: ₹30,000 – ₹80,000\n⏳ Duration: 1–3 years"
    elif mode == "Momentum":
        return "💰 Suggested Investment: ₹20,000 – ₹50,000\n⏳ Duration: 1–6 months"
    elif mode == "Balanced":
        return "💰 Suggested Investment: ₹40,000 – ₹70,000\n⏳ Duration: 1–2 years"
    return "💰 Suggested Investment: Flexible\n⏳ Duration: Depends on strategy"

# ------------------------------------------------------
# MAIN APP
# ------------------------------------------------------
st.title("📈 AI-Based Share Market Recommendation Tool")

symbols = st.text_input("Enter stock symbols (comma separated):", "RELIANCE.NS, TCS.NS, INFY.NS")
ranking_mode = st.selectbox("Choose Ranking Mode", ["Growth", "Value", "Momentum", "Balanced"])

st.info(f"📖 Explanation of {ranking_mode} Strategy: {ranking_explanations[ranking_mode]}")

symbols = [s.strip() for s in symbols.split(",")]
stock_data = fetch_stock_data(symbols)

stock_features = {}
for sym, df in stock_data.items():
    stock_features[sym] = compute_features(df)

rows = []
for sym, df in stock_features.items():
    if df.empty: 
        continue
    latest = df.iloc[-1]
    score = calculate_score(latest, ranking_mode)
    rows.append({
        "Company Name": sym,
        "Symbol": sym,
        "Last": latest["Close"],
        "Composite Score": score,
        "ret_60": latest.get("ret_60", np.nan),
        "pct_to_52w_high": latest.get("pct_to_52w_high", np.nan),
        "rsi14": latest.get("rsi14", np.nan)
    })

suggestions = pd.DataFrame(rows).sort_values(by="Composite Score", ascending=False).head(5)

if not suggestions.empty:
    st.subheader("📊 Top Stock Suggestions")
    st.dataframe(suggestions[["Company Name", "Symbol", "Last", "Composite Score", "ret_60", "pct_to_52w_high", "rsi14"]], use_container_width=True)

    st.subheader("📌 Detailed Suggestions (Easy to Understand)")
    cols = st.columns(2)
    for i, (_, row) in enumerate(suggestions.iterrows()):
        with cols[i % 2]:
            sym = row["Symbol"]
            df = stock_data[sym]
            sparkline = create_sparkline(df)
            st.markdown(f"""
            ### {row['Company Name']}
            **Current Price:** ₹{row['Last']:.2f}  
            **Reason (Simple):** {get_reason(row, ranking_mode)}  
            **Plan:** {investment_plan(row, ranking_mode)}  

            <img src="{sparkline}" width="100%" />
            """, unsafe_allow_html=True)
else:
    st.warning("No valid stock data found. Please check symbols.")
