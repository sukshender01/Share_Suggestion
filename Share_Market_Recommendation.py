# streamlit_app.py
# -------------------------------------------------------------
# India Stocks: Real-time Gainers/Losers + Smart Suggestions
# -------------------------------------------------------------
# HOW TO RUN
# 1) pip install -r requirements.txt  (see requirements list below)
# 2) streamlit run streamlit_app.py
#
# NOTES
# - Data source for index constituents: Nifty Indices CSV (official) 
#   NIFTY 50:     https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv
#   NIFTY 500:    https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv
# - Live/near-real-time prices fetched from Yahoo Finance via yfinance (delayed).
# - Predictions are educational only, not investment advice.
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime
from functools import lru_cache
from typing import List

st.set_page_config(
    page_title="India Stocks Screener & Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.title("⚙️ Controls")
universe_choice = st.sidebar.selectbox(
    "Universe",
    ["NIFTY 50", "NIFTY 500"],
    index=1,
    help="Choose index universe for scanning.",
)
refresh_minutes = st.sidebar.slider("Auto-refresh (minutes)", 0, 10, 1, help="0 disables auto-refresh")
rank_by = st.sidebar.selectbox(
    "Rank predictions by",
    ["Composite Score", "3M Momentum", "Near 52W High", "RSI (Oversold First)"]
)
suggestions_count = st.sidebar.slider("# Suggestions", 5, 30, 10)
risk_pref = st.sidebar.select_slider("Risk preference", options=["Low", "Medium", "High"], value="Medium")

if refresh_minutes > 0:
    st.experimental_set_query_params(_=str(int(datetime.utcnow().timestamp()) // (refresh_minutes*60 if refresh_minutes else 1)))

# ----------------------------
# Helpers
# ----------------------------
NIFTY_50_CSV = "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"
NIFTY_500_CSV = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"

@lru_cache(maxsize=8)
def fetch_index_constituents(url: str) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    if "Symbol" in df.columns:
        df["Yahoo"] = df["Symbol"].str.strip() + ".NS"
    elif "SYMBOL" in df.columns:
        df["Yahoo"] = df["SYMBOL"].str.strip() + ".NS"
    return df

@st.cache_data(ttl=60*15)
def get_universe(choice: str) -> pd.DataFrame:
    if choice == "NIFTY 50":
        return fetch_index_constituents(NIFTY_50_CSV)
    return fetch_index_constituents(NIFTY_500_CSV)

@st.cache_data(ttl=60)
def batch_latest_quote(yahoo_symbols: List[str]) -> pd.DataFrame:
    tickers = " ".join(yahoo_symbols)
    data = yf.download(tickers=tickers, period="2d", interval="1d", group_by="ticker", auto_adjust=False, threads=True, progress=False)

    rows = []
    if isinstance(data.columns, pd.MultiIndex):
        for sym in yahoo_symbols:
            try:
                df = data[sym]
                if df.shape[0] >= 2:
                    prev_close = float(df.iloc[-2]["Close"])
                    last = float(df.iloc[-1]["Close"])
                    pct = (last - prev_close) / prev_close * 100.0
                    rows.append((sym, last, prev_close, pct))
            except Exception:
                continue
    quotes = pd.DataFrame(rows, columns=["Yahoo", "Last", "PrevClose", "%Chg"])
    return quotes

@st.cache_data(ttl=60*60)
def get_history(yahoo_symbols: List[str], period="2y") -> dict:
    hist = {}
    batch = 50
    for i in range(0, len(yahoo_symbols), batch):
        chunk = yahoo_symbols[i:i+batch]
        df = yf.download(" ".join(chunk), period=period, interval="1d", group_by="ticker", auto_adjust=True, threads=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            for sym in chunk:
                try:
                    h = df[sym].dropna()
                    if not h.empty:
                        hist[sym] = h
                except Exception:
                    continue
    return hist

# Technicals

def compute_indicators(price: pd.DataFrame) -> pd.DataFrame:
    x = price.copy()
    c = x["Close"].astype(float)
    x["ret_60"] = c.pct_change(60)
    x["rsi14"] = 100 - (100 / (1 + (c.diff().clip(lower=0).rolling(14).mean() / (-c.diff().clip(upper=0)).rolling(14).mean().replace(0, np.nan))))
    x["roll_max_252"] = c.rolling(252).max()
    x["pct_to_52w_high"] = (c / x["roll_max_252"]) - 1
    return x

def composite_score(latest: pd.Series, risk: str) -> float:
    score = 0.0
    score += 1.0 * (latest.get("ret_60", 0) or 0)
    score += 1.25 * (latest.get("pct_to_52w_high", -1) or -1)
    return float(score)

# ----------------------------
# Data Pipeline
# ----------------------------
universe_df = get_universe(universe_choice)
st.title("🇮🇳 India Stocks Screener & Predictor")
st.caption("Top gainers/losers (near-real-time) and data-driven suggestions. ✦ Educational use only, not investment advice.")

# Movers
symbols = universe_df["Yahoo"].dropna().unique().tolist()
quotes = batch_latest_quote(symbols)
merged = universe_df.merge(quotes, on="Yahoo", how="inner")

gainers = merged.sort_values("%Chg", ascending=False).head(20)
losers = merged.sort_values("%Chg", ascending=True).head(20)

c1, c2 = st.columns(2)
with c1:
    st.write("**Top 20 Gainers**")
    st.dataframe(gainers[["Company Name", "Symbol", "Last", "%Chg"]].set_index("Symbol"))
with c2:
    st.write("**Top 20 Losers**")
    st.dataframe(losers[["Company Name", "Symbol", "Last", "%Chg"]].set_index("Symbol"))

st.divider()

# Predictions / Suggestions
st.subheader("🤖 Smart Suggestions")

hist = get_history(symbols, period="2y")
rows = []
for sym, h in hist.items():
    ind = compute_indicators(h)
    if ind.empty: continue
    latest = ind.iloc[-1]
    score = composite_score(latest, risk_pref)
    rows.append({
        "Symbol": sym.replace(".NS", ""),
        "Yahoo": sym,
        "Last": float(ind["Close"].iloc[-1]),
        "ret_60": float(latest.get("ret_60", np.nan)),
        "rsi14": float(latest.get("rsi14", np.nan)),
        "pct_to_52w_high": float(latest.get("pct_to_52w_high", np.nan)),
        "Composite Score": score,
    })

pred_df = pd.DataFrame(rows)
if rank_by == "3M Momentum":
    pred_df = pred_df.sort_values("ret_60", ascending=False)
elif rank_by == "Near 52W High":
    pred_df = pred_df.sort_values("pct_to_52w_high", ascending=False)
elif rank_by == "RSI (Oversold First)":
    pred_df = pred_df.sort_values("rsi14", ascending=True)
else:
    pred_df = pred_df.sort_values("Composite Score", ascending=False)

suggestions = pred_df.head(suggestions_count)
suggestions = suggestions.merge(universe_df[["Symbol", "Company Name", "Yahoo"]], on="Yahoo", how="left")

st.write("**Suggested candidates (not advice):**")
st.dataframe(
    suggestions[["Company Name", "Symbol", "Last", "Composite Score", "ret_60", "pct_to_52w_high", "rsi14"]]
    .set_index("Symbol")
    .style.format({"Last": "₹{:.2f}", "Composite Score": "{:.3f}", "ret_60": "{:.2%}", "pct_to_52w_high": "{:.2%}", "rsi14": "{:.1f}"})
)

# 3M Momentum Chart if selected
if rank_by == "3M Momentum" and not suggestions.empty:
    st.subheader("📈 3-Month Return Trendlines for Top Picks")
    import plotly.express as px
    for sym in suggestions["Yahoo"].tolist():
        h = hist.get(sym)
        if h is not None and not h.empty:
            h3m = h.tail(90).reset_index()
            h3m["Return"] = h3m["Close"].pct_change().cumsum()
            fig = px.line(h3m, x="Date", y="Return", title=f"{sym.replace('.NS','')} - 3M Cumulative Return")
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.caption("Data via Yahoo Finance (delayed). Predictions are heuristic, for research/education only.")

# ----------------------------
# REQUIREMENTS (requirements.txt)
# ----------------------------
# streamlit>=1.36
# yfinance>=0.2.40
# pandas>=2.2
# numpy>=1.26
# plotly>=5.22
# requests>=2.31
