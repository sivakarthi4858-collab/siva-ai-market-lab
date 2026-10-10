
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="SIVA AI Market Lab",
    page_icon="📈",
    layout="wide"
)

st.title("📈 SIVA AI Market Lab")
st.caption("OTC Research • Demo Only • No Live Trading")

st.warning(
    "Educational tool only. No guaranteed profits. "
    "No broker connection or automatic orders."
)

with st.sidebar:
    st.header("Analysis Settings")
    market = st.text_input("Market / source label", "Quotex OTC")
    fast = st.number_input("Fast EMA", 2, 50, 9)
    slow = st.number_input("Slow EMA", 3, 200, 21)
    rsi_period = st.number_input("RSI period", 2, 50, 14)
    hold = st.number_input("Backtest hold (candles)", 1, 20, 1)

st.subheader("1. Upload real historical data")
st.write("CSV columns required: timestamp, open, high, low, close")
uploaded = st.file_uploader("Choose OHLC CSV file", type=["csv"])

st.info(
    "Data source limitation: Quotex-ku official public API illa. "
    "OTCharts free tier-la USD/INR OTC available-nu confirm aagala. "
    "Verified matching data illana indha app signals kaattaadhu. "
    "Different market data-ai USD/INR OTC-nu label panna vendaam."
)
st.link_button("Check OTCharts supported symbols", "https://otcharts.com/quotex-api")

if uploaded is None:
    st.subheader("2. Signal dashboard")
    st.write("CSV upload pannina piragu dhaan analysis varum.")
    st.write("**Signal:** WAITING FOR VERIFIED DATA")
    st.caption(
        "Example/sample candles use pannala. Data illama signal "
        "generate pannradhu thavirkappadum."
    )
    st.stop()

try:
    df = pd.read_csv(uploaded)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        st.error("Missing CSV columns: " + ", ".join(missing))
        st.stop()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=required)
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    if len(df) < max(int(fast), int(slow), int(rsi_period)) + 5:
        st.error("More valid historical candles upload pannu.")
        st.stop()

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        st.error("OHLC prices positive-a irukkanum.")
        st.stop()

    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any():
        st.error("Invalid candle: High value check pannu.")
        st.stop()

    if (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
        st.error("Invalid candle: Low value check pannu.")
        st.stop()

except Exception as error:
    st.error(f"CSV read error: {error}")
    st.stop()

df["ema_fast"] = df["close"].ewm(span=int(fast), adjust=False).mean()
df["ema_slow"] = df["close"].ewm(span=int(slow), adjust=False).mean()

change = df["close"].diff()
gain = change.clip(lower=0)
loss = -change.clip(upper=0)
avg_gain = gain.ewm(
    alpha=1 / int(rsi_period),
    min_periods=int(rsi_period),
    adjust=False
).mean()
avg_loss = loss.ewm(
    alpha=1 / int(rsi_period),
    min_periods=int(rsi_period),
    adjust=False
).mean()

rs = avg_gain / avg_loss.replace(0, np.nan)
df["rsi"] = 100 - (100 / (1 + rs))

df["signal"] = np.select(
    [
        (df["ema_fast"] > df["ema_slow"]) & (df["rsi"] > 50),
        (df["ema_fast"] < df["ema_slow"]) & (df["rsi"] < 50)
    ],
    ["BUY", "SELL"],
    default="WAIT"
)

st.subheader("3. Latest analysis")
latest = df.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Close", f"{latest['close']:.6g}")
c2.metric("Signal", latest["signal"])
c3.metric(f"EMA {int(fast)}", f"{latest['ema_fast']:.6g}")
c4.metric(
    "RSI",
    "Not ready" if pd.isna(latest["rsi"]) else f"{latest['rsi']:.1f}"
)
st.caption(f"Source label entered: {market}. Source identity is not independently verified.")

st.subheader("4. Candlestick chart")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["timestamp"],
    open=df["open"], high=df["high"],
    low=df["low"], close=df["close"],
    name="OHLC"
))
fig.add_trace(go.Scatter(
    x=df["timestamp"], y=df["ema_fast"], name="Fast EMA"
))
fig.add_trace(go.Scatter(
    x=df["timestamp"], y=df["ema_slow"], name="Slow EMA"
))
fig.update_layout(height=500, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("5. Historical directional backtest")
test = df.copy()
test["future_close"] = test["close"].shift(-int(hold))
test = test.dropna(subset=["future_close"])
test = test[test["signal"].isin(["BUY", "SELL"])].copy()

if test.empty:
    st.info("Backtest panna signals illa. More historical data try pannu.")
else:
    test["move_pct"] = (
        (test["future_close"] / test["close"]) - 1
    ) * 100
    test["direction_correct"] = np.where(
        test["signal"] == "BUY",
        test["move_pct"] > 0,
        test["move_pct"] < 0
    )
    hit_rate = test["direction_correct"].mean() * 100
    c1, c2 = st.columns(2)
    c1.metric("Signals tested", len(test))
    c2.metric("Historical direction match", f"{hit_rate:.1f}%")
    st.dataframe(
        test[[
            "timestamp", "close", "signal",
            "future_close", "move_pct", "direction_correct"
        ]].tail(100),
        use_container_width=True
    )

st.caption(
    "Backtest is candle-direction only, not Quotex payout/expiry simulation. "
    "It excludes fees, ties, execution delays and payout rules. "
    "Historical results do not predict future results."
)
st.caption("No real trades placed. Demo/research only.")
        
