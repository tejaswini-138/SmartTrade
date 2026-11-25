import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
from datetime import date, datetime, timedelta

st.set_page_config(page_title="SmartTrade AI", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
[data-testid="stMetric"] {background: rgba(70,90,120,.12); border: 1px solid rgba(120,150,180,.18); padding: 14px; border-radius: 14px;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=900)
def load_data(symbol, period):
    try:
        df = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename_axis(None)

        required = ["Open", "High", "Low", "Close", "Volume"]

        if not all(col in df.columns for col in required):
            return pd.DataFrame()

        df = df[required].copy()
        df = df.dropna()

        return df

    except Exception as e:
        st.error(f"Data loading error for {symbol}: {e}")
        return pd.DataFrame()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(df):
    d = df.copy()
    close = d["Close"]
    d["SMA_10"] = close.rolling(10).mean()
    d["SMA_20"] = close.rolling(20).mean()
    d["RSI_14"] = rsi(close)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["Return_1d"] = close.pct_change()
    d["Return_5d"] = close.pct_change(5)
    d["Volume_Change"] = d["Volume"].pct_change()
    d["Target_Close"] = close.shift(-1)
    d["Target_Direction"] = (d["Target_Close"] > close).astype(int)
    return d.replace([np.inf, -np.inf], np.nan).dropna()


st.title("📈 SmartTrade AI")
# ================= MARKET OVERVIEW =================
@st.cache_data(ttl=300)
def get_market_data(ticker):
    try:
        # Method 1: Download using yfinance
        data = yf.download(
            ticker,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False
        )

        # Method 2: Fallback if download fails
        if data is None or data.empty:
            data = yf.Ticker(ticker).history(period="5d")

        if data is None or data.empty:
            print(f"Data unavailable for {ticker}")
            return None, None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        close = data["Close"].dropna()

        if len(close) < 2:
            return None, None

        current = float(close.iloc[-1])
        previous = float(close.iloc[-2])

        change = ((current - previous) / previous) * 100

        return current, change

    except Exception as e:
        print(f"Market data error for {ticker}: {e}")
        return None, None


st.subheader("🌍 Market Overview")

nifty_value, nifty_change = get_market_data("^NSEI")
sensex_value, sensex_change = get_market_data("^BSESN")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "NIFTY 50",
        f"{nifty_value:,.2f}" if nifty_value is not None else "Unavailable",
        f"{nifty_change:+.2f}%" if nifty_change is not None else None
    )

with col2:
    st.metric(
        "SENSEX",
        f"{sensex_value:,.2f}" if sensex_value is not None else "Unavailable",
        f"{sensex_change:+.2f}%" if sensex_change is not None else None
    )

with col3:
    if nifty_change is not None:
        if nifty_change > 0:
            market_trend = "🟢 Bullish"
        elif nifty_change < 0:
            market_trend = "🔴 Bearish"
        else:
            market_trend = "🟡 Neutral"
    else:
        market_trend = "⚪ Unavailable"

    st.metric("Market Sentiment", market_trend)

st.divider()
with st.sidebar:
    st.header("Stock Selection")
    # Stock selection by sector
stock_sectors = {
    "IT & Technology": [
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"
    ],
    "Banking & Finance": [
        "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK",
        "KOTAKBANK", "BAJFINANCE", "PNB"
    ],
    "Energy & Power": [
        "RELIANCE", "NTPC", "POWERGRID", "ONGC", "COALINDIA"
    ],
    "Automobile": [
        "MARUTI", "M&M", "TATAMOTORS"
    ],
    "Pharma & Healthcare": [
        "SUNPHARMA", "CIPLA", "DRREDDY"
    ],
    "FMCG & Consumer": [
        "ITC", "HINDUNILVR", "ASIANPAINT", "TITAN", "TRENT"
    ],
    "Infrastructure & Industrial": [
        "LT", "ADANIENT", "ADANIPORTS", "JSWSTEEL",
        "TATASTEEL", "ULTRACEMCO", "BEL"
    ]
}

selected_sector = st.sidebar.selectbox(
    "🏭 Select Sector",
    list(stock_sectors.keys())
)

stock_name = st.sidebar.selectbox(
    "📈 Select Stock",
    stock_sectors[selected_sector]
)

symbol = stock_name + ".NS"
if "last_refresh_time" not in st.session_state:
    st.session_state.last_refresh_time = datetime.now()
period = st.selectbox("Historical data range", ["6mo", "1y", "2y", "5y"], index=1)
st.info("NSE ticker symbols are used with Yahoo Finance data. Availability may vary.")
refresh = st.sidebar.button("🔄 Refresh Data")

if refresh:
    load_data.clear()
    get_market_data.clear()
    st.session_state.last_refresh_time = datetime.now()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🕒 Dashboard Status")

last_updated = st.session_state.last_refresh_time.strftime(
    "%d %b %Y • %I:%M:%S %p"
)

st.sidebar.success(f"Last refreshed: {last_updated}")
with st.spinner("Loading historical market data..."):
    raw = load_data(symbol, period)

if raw.empty:
    st.error("Market data could not be retrieved right now. Check your internet connection or try another ticker later.")
    st.stop()

featured = add_features(raw)
if len(featured) < 80:
    st.warning("Not enough rows after indicator calculations to train a reliable model. Try a longer historical range.")
    st.stop()

latest = raw.iloc[-1]
prev_close = float(raw["Close"].iloc[-1])
change = float(raw["Close"].iloc[-1] - raw["Close"].iloc[-2])
change_pct = change / float(raw["Close"].iloc[-2]) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{stock_name} Latest Close", f"₹{prev_close:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
c2.metric("SMA 10", f"₹{float(raw['Close'].rolling(10).mean().iloc[-1]):,.2f}")
c3.metric("SMA 20", f"₹{float(raw['Close'].rolling(20).mean().iloc[-1]):,.2f}")
c4.metric("RSI (14)", f"{float(rsi(raw['Close']).iloc[-1]):.1f}")

st.subheader("Historical Price")
chart = go.Figure()
chart.add_trace(go.Scatter(x=raw.index, y=raw["Close"], name="Close", line=dict(width=2)))
chart.add_trace(go.Scatter(x=raw.index, y=raw["Close"].rolling(10).mean(), name="SMA 10", line=dict(width=1.5)))
chart.add_trace(go.Scatter(x=raw.index, y=raw["Close"].rolling(20).mean(), name="SMA 20", line=dict(width=1.5)))
chart.update_layout(height=430, template="plotly_dark", hovermode="x unified", margin=dict(l=10,r=10,t=25,b=10), legend=dict(orientation="h", y=1.02))
st.plotly_chart(chart, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Technical Indicators")
    ind = pd.DataFrame({"RSI (14)": rsi(raw["Close"]), "MACD": raw["Close"].ewm(span=12, adjust=False).mean() - raw["Close"].ewm(span=26, adjust=False).mean()})
    st.line_chart(ind.dropna(), height=240)
with right:
    st.subheader("Recent Market Data")
    st.dataframe(raw.tail(8).sort_index(ascending=False)[["Open","High","Low","Close","Volume"]], use_container_width=True)

st.divider()
st.subheader("📈 Trend-Based Prediction")
st.write("Estimate the next closing price using a simple least-squares linear trend over recent closing prices. This lightweight fallback avoids the blocked SciPy/scikit-learn DLLs on this Windows system.")
if st.button("✨ Run prediction", type="primary", use_container_width=True):
    with st.spinner("Fitting trend and evaluating holdout..."):
        closes = [float(v) for v in raw["Close"].dropna().tolist()]
        split = int(len(closes) * 0.8)
        test_start = max(split, 30)
        window = min(60, test_start)

        def fit_line_predict(values, horizon=1):
            vals = [float(v) for v in values[-window:]]
            n = len(vals)
            xs = list(range(n)
            )
            mean_x = sum(xs) / n
            mean_y = sum(vals) / n
            denom = sum((x - mean_x) ** 2 for x in xs)
            slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, vals)) / denom if denom else 0.0
            intercept = mean_y - slope * mean_x
            return intercept + slope * (n - 1 + horizon)

        forecast = fit_line_predict(closes, 1)
                # -------- BUY / HOLD / SELL RECOMMENDATION --------

        sma10_now = float(raw["Close"].rolling(10).mean().iloc[-1])
        sma20_now = float(raw["Close"].rolling(20).mean().iloc[-1])

        rsi_now = float(rsi(raw["Close"]).iloc[-1])

        ema12_now = raw["Close"].ewm(span=12, adjust=False).mean().iloc[-1]
        ema26_now = raw["Close"].ewm(span=26, adjust=False).mean().iloc[-1]

        macd_now = float(ema12_now - ema26_now)

        macd_series = (
            raw["Close"].ewm(span=12, adjust=False).mean()
            - raw["Close"].ewm(span=26, adjust=False).mean()
        )

        macd_signal_now = float(
            macd_series.ewm(span=9, adjust=False).mean().iloc[-1]
        )

        score = 0
        reasons = []

        # 1. SMA trend
        if sma10_now > sma20_now:
            score += 1
            reasons.append("Bullish: SMA 10 is above SMA 20.")
        else:
            score -= 1
            reasons.append("Bearish: SMA 10 is below SMA 20.")

        # 2. MACD momentum
        if macd_now > macd_signal_now:
            score += 1
            reasons.append("Bullish: MACD is above its signal line.")
        else:
            score -= 1
            reasons.append("Bearish: MACD is below its signal line.")

        # 3. RSI momentum
        if 50 <= rsi_now <= 70:
            score += 1
            reasons.append(f"Positive momentum: RSI is {rsi_now:.1f}.")
        elif 30 <= rsi_now < 50:
            score -= 1
            reasons.append(f"Weaker momentum: RSI is {rsi_now:.1f}.")
        elif rsi_now > 70:
            score -= 1
            reasons.append(f"Warning: RSI {rsi_now:.1f} is in an overbought zone.")
        else:
            reasons.append(f"RSI is {rsi_now:.1f}; oversold zone, treated as neutral.")

        # 4. Predicted price trend
        if forecast > closes[-1]:
            score += 1
            reasons.append("Bullish: predicted price is above the latest close.")
        elif forecast < closes[-1]:
            score -= 1
            reasons.append("Bearish: predicted price is below the latest close.")
        else:
            reasons.append("Neutral: predicted price is unchanged.")

        # Final recommendation
        if score >= 2:
            recommendation = "BUY"
            color = "#16A34A"
            emoji = "🟢"
            summary = "Positive signals dominate the current analysis."
        elif score <= -2:
            recommendation = "SELL"
            color = "#DC2626"
            emoji = "🔴"
            summary = "Negative signals dominate the current analysis."
        else:
            recommendation = "HOLD"
            color = "#D97706"
            emoji = "🟡"
            summary = "Signals are mixed or not strong enough."

        # Display recommendation card
        st.markdown("---")
        st.subheader("🤖 AI Stock Recommendation")

        st.markdown(
            f"""
            <div style="
                background-color:{color}18;
                border:2px solid {color};
                border-radius:15px;
                padding:22px;
                text-align:center;
            ">
                <h1 style="color:{color}; margin:0;">
                    {emoji} {recommendation}
                </h1>
                <h3>Signal Score: {score:+d} / 4</h3>
                <p>{summary}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("📋 Why this recommendation?")

        for reason in reasons:
            st.write("•", reason)

        st.caption(
            "This recommendation is generated from technical rules and a "
            "trend forecast. It is not a guarantee of future stock performance "
            "or financial advice."
        )
        # Rolling one-step trend forecasts on the holdout period (past-only inputs).
        test_actual = closes[test_start:]
        test_pred = []
        for idx in range(test_start, len(closes)):
            test_pred.append(fit_line_predict(closes[:idx], 1))
        if test_actual and test_pred:
            errors = [a - p for a, p in zip(test_actual, test_pred)]
            mae = sum(abs(e) for e in errors) / len(errors)
            rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
            correct = sum((test_pred[i] > closes[test_start+i-1]) == (test_actual[i] > closes[test_start+i-1]) for i in range(len(test_actual)))
            direction_accuracy = correct / len(test_actual)
        else:
            mae = rmse = direction_accuracy = 0.0
        trend = "UP" if forecast > closes[-1] else "DOWN / FLAT"
        p1, p2, p3 = st.columns(3)
        p1.metric("Estimated Next Close", f"₹{forecast:,.2f}", f"{forecast-closes[-1]:+,.2f} vs latest close")
        p2.metric("Estimated Direction", trend)
        p3.metric("Holdout Direction Accuracy", f"{direction_accuracy*100:.1f}%")
        st.caption(f"Chronological holdout evaluation — MAE: ₹{mae:.2f} · RMSE: ₹{rmse:.2f}. This is a trend baseline, not a Random Forest model.")
        pred_fig = go.Figure()
        pred_fig.add_trace(go.Scatter(x=raw.index[-60:], y=raw["Close"].tail(60), name="Recent close", line=dict(width=2)))
        next_day = raw.index[-1] + pd.Timedelta(days=1)
        pred_fig.add_trace(go.Scatter(x=[raw.index[-1], next_day], y=[closes[-1], forecast], name="Trend estimate", mode="lines+markers", line=dict(width=3, dash="dash")))
        pred_fig.update_layout(height=350, template="plotly_dark", hovermode="x unified", margin=dict(l=10,r=10,t=25,b=10))
        st.plotly_chart(pred_fig, use_container_width=True)
        st.warning("Prediction is a trend estimate for educational use, not investment advice. The next calendar day shown may be a non-trading day.")

st.divider()
st.caption("Built for an academic AI/ML project · Market data via Yahoo Finance (yfinance) · No real trades are placed")
