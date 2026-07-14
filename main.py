import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

st.set_page_config(page_title="📈 주식 데이터 분석", page_icon="📈", layout="wide")


# ---------- 데이터 로딩 ----------
@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_company_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df


def build_chart(df: pd.DataFrame, ticker: str, show_sma: list, show_rsi: bool, show_macd: bool) -> go.Figure:
    rows = 2 + int(show_rsi) + int(show_macd)
    row_heights = [0.5, 0.15]
    specs_titles = ["가격", "거래량"]
    if show_rsi:
        row_heights.append(0.15)
        specs_titles.append("RSI")
    if show_macd:
        row_heights.append(0.2)
        specs_titles.append("MACD")

    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=row_heights,
        subplot_titles=specs_titles,
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=ticker, increasing_line_color="#e74c3c", decreasing_line_color="#3498db",
    ), row=1, col=1)

    colors = {"SMA20": "#f39c12", "SMA50": "#9b59b6", "SMA200": "#2ecc71"}
    for sma in show_sma:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[sma], name=sma, line=dict(width=1.3, color=colors.get(sma)),
        ), row=1, col=1)

    vol_colors = ["#e74c3c" if c >= o else "#3498db" for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="거래량", marker_color=vol_colors, showlegend=False), row=2, col=1)

    next_row = 3
    if show_rsi:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#1abc9c")), row=next_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=next_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=next_row, col=1)
        next_row += 1

    if show_macd:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#e67e22")), row=next_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="#34495e")), row=next_row, col=1)

    fig.update_layout(
        height=750, template="plotly_white", xaxis_rangeslider_visible=False,
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ---------- 사이드바 ----------
st.sidebar.title("⚙️ 설정")
ticker = st.sidebar.text_input("종목 티커 (예: AAPL, TSLA, 005930.KS)", value="AAPL").strip().upper()

period_option = st.sidebar.selectbox(
    "기간", ["1개월", "3개월", "6개월", "1년", "3년", "5년", "직접 선택"], index=3,
)
period_map = {"1개월": 30, "3개월": 90, "6개월": 182, "1년": 365, "3년": 365 * 3, "5년": 365 * 5}

if period_option == "직접 선택":
    start_date = st.sidebar.date_input("시작일", value=date.today() - timedelta(days=365))
    end_date = st.sidebar.date_input("종료일", value=date.today())
else:
    end_date = date.today()
    start_date = end_date - timedelta(days=period_map[period_option])

st.sidebar.markdown("**보조 지표**")
sma_options = st.sidebar.multiselect("이동평균선", ["SMA20", "SMA50", "SMA200"], default=["SMA20", "SMA50"])
show_rsi = st.sidebar.checkbox("RSI 표시", value=True)
show_macd = st.sidebar.checkbox("MACD 표시", value=False)

st.title("📈 인터랙티브 주식 데이터 분석")
st.caption("데이터 출처: Yahoo Finance (yfinance)")

if not ticker:
    st.info("왼쪽 사이드바에 종목 티커를 입력해주세요.")
    st.stop()

with st.spinner(f"{ticker} 데이터를 불러오는 중..."):
    raw_df = load_price_data(ticker, start_date, end_date)

if raw_df.empty:
    st.error("데이터를 불러올 수 없습니다. 티커를 확인해주세요. (예: 삼성전자 → 005930.KS)")
    st.stop()

df = add_indicators(raw_df)
info = load_company_info(ticker)

# ---------- 상단 지표 카드 ----------
last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
change = last_close - prev_close
change_pct = (change / prev_close * 100) if prev_close else 0

col1, col2, col3, col4, col5 = st.columns(5)
company_name = info.get("longName", ticker)
col1.metric("종목", company_name if len(company_name) < 20 else ticker)
col2.metric("현재가", f"{last_close:,.2f}", f"{change:+,.2f} ({change_pct:+.2f}%)")
col3.metric("52주 최고", f"{info.get('fiftyTwoWeekHigh', df['High'].max()):,.2f}")
col4.metric("52주 최저", f"{info.get('fiftyTwoWeekLow', df['Low'].min()):,.2f}")
mcap = info.get("marketCap")
col5.metric("시가총액", f"{mcap / 1e8:,.0f}억" if mcap else "N/A")

st.divider()

# ---------- 메인 차트 ----------
fig = build_chart(df, ticker, sma_options, show_rsi, show_macd)
st.plotly_chart(fig, use_container_width=True)

# ---------- 데이터 테이블 & 다운로드 ----------
with st.expander("📋 원본 데이터 보기"):
    st.dataframe(df.sort_index(ascending=False), use_container_width=True)
    csv = df.to_csv().encode("utf-8-sig")
    st.download_button("CSV 다운로드", data=csv, file_name=f"{ticker}_data.csv", mime="text/csv")

# ---------- 기업 개요 ----------
if info.get("longBusinessSummary"):
    with st.expander("🏢 기업 개요"):
        st.write(info["longBusinessSummary"])
