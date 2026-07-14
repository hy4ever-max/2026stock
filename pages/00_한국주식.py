import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

st.set_page_config(page_title="🇰🇷 AI·반도체 대표주 분석", page_icon="🔬", layout="wide")

# ---------- 한국 AI·반도체 대표주 목록 ----------
# .KS = 코스피, .KQ = 코스닥
STOCK_UNIVERSE = {
    "반도체": {
        "삼성전자": "005930.KS",
        "SK하이닉스": "000660.KS",
        "DB하이텍": "000990.KS",
        "한미반도체": "042700.KS",
        "리노공업": "058470.KQ",
        "이수페타시스": "007660.KS",
        "솔브레인": "036830.KQ",
        "원익IPS": "240810.KQ",
        "티씨케이": "064760.KQ",
        "동진쎄미켐": "005290.KQ",
        "심텍": "222800.KQ",
        "ISC": "095340.KQ",
    },
    "AI / 플랫폼 / SW": {
        "네이버": "035420.KS",
        "카카오": "035720.KS",
        "삼성SDS": "018260.KS",
        "더존비즈온": "012510.KQ",
        "솔트룩스": "304100.KQ",
        "셀바스AI": "108860.KQ",
        "코난테크놀로지": "402030.KQ",
        "알체라": "347860.KQ",
    },
    "AI 서버 / 전력·부품": {
        "LG이노텍": "011070.KS",
        "삼성전기": "009150.KS",
        "HPSP": "403870.KQ",
        "이오테크닉스": "039030.KQ",
        "미래컴퍼니": "049950.KQ",
    },
}

ALL_STOCKS = {name: code for group in STOCK_UNIVERSE.values() for name, code in group.items()}


# ---------- 데이터 로딩 ----------
@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


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


def build_chart(df: pd.DataFrame, name: str, show_sma: list, show_rsi: bool, show_macd: bool) -> go.Figure:
    rows = 2 + int(show_rsi) + int(show_macd)
    row_heights = [0.5, 0.15]
    titles = ["가격", "거래량"]
    if show_rsi:
        row_heights.append(0.15)
        titles.append("RSI")
    if show_macd:
        row_heights.append(0.2)
        titles.append("MACD")
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                         row_heights=row_heights, subplot_titles=titles)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=name, increasing_line_color="#e74c3c", decreasing_line_color="#3498db",
    ), row=1, col=1)

    colors = {"SMA20": "#f39c12", "SMA50": "#9b59b6", "SMA200": "#2ecc71"}
    for sma in show_sma:
        fig.add_trace(go.Scatter(x=df.index, y=df[sma], name=sma,
                                  line=dict(width=1.3, color=colors.get(sma))), row=1, col=1)

    vol_colors = ["#e74c3c" if c >= o else "#3498db" for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="거래량",
                          marker_color=vol_colors, showlegend=False), row=2, col=1)

    next_row = 3
    if show_rsi:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#1abc9c")), row=next_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=next_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=next_row, col=1)
        next_row += 1

    if show_macd:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#e67e22")), row=next_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="#34495e")), row=next_row, col=1)

    fig.update_layout(height=750, template="plotly_white", xaxis_rangeslider_visible=False,
                       hovermode="x unified",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                       margin=dict(l=10, r=10, t=40, b=10))
    return fig


def build_comparison_chart(price_dict: dict) -> go.Figure:
    fig = go.Figure()
    for name, series in price_dict.items():
        normalized = series / series.iloc[0] * 100
        fig.add_trace(go.Scatter(x=normalized.index, y=normalized, name=name, mode="lines"))
    fig.update_layout(
        title="종목별 상대 수익률 비교 (시작일 = 100)", template="plotly_white",
        height=500, hovermode="x unified", yaxis_title="지수 (시작일=100)",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ---------- 사이드바 ----------
st.sidebar.title("⚙️ 설정")

period_option = st.sidebar.selectbox("기간", ["1개월", "3개월", "6개월", "1년", "3년", "5년", "직접 선택"], index=3)
period_map = {"1개월": 30, "3개월": 90, "6개월": 182, "1년": 365, "3년": 365 * 3, "5년": 365 * 5}

if period_option == "직접 선택":
    start_date = st.sidebar.date_input("시작일", value=date.today() - timedelta(days=365))
    end_date = st.sidebar.date_input("종료일", value=date.today())
else:
    end_date = date.today()
    start_date = end_date - timedelta(days=period_map[period_option])

st.title("🔬 한국 AI · 반도체 대표주 분석")
st.caption("데이터 출처: Yahoo Finance (yfinance) · 코스피(.KS) / 코스닥(.KQ)")

tab1, tab2 = st.tabs(["📊 개별 종목 분석", "⚖️ 종목 비교"])

# ---------- 탭 1: 개별 종목 ----------
with tab1:
    group = st.selectbox("업종 그룹", list(STOCK_UNIVERSE.keys()))
    stock_name = st.selectbox("종목 선택", list(STOCK_UNIVERSE[group].keys()))
    ticker = STOCK_UNIVERSE[group][stock_name]

    st.markdown("**보조 지표**")
    c1, c2, c3 = st.columns(3)
    sma_options = c1.multiselect("이동평균선", ["SMA20", "SMA50", "SMA200"], default=["SMA20", "SMA50"])
    show_rsi = c2.checkbox("RSI 표시", value=True)
    show_macd = c3.checkbox("MACD 표시", value=False)

    with st.spinner(f"{stock_name} ({ticker}) 데이터를 불러오는 중..."):
        raw_df = load_price_data(ticker, start_date, end_date)

    if raw_df.empty:
        st.error("데이터를 불러올 수 없습니다. 잠시 후 다시 시도해주세요.")
    else:
        df = add_indicators(raw_df)
        info = load_company_info(ticker)

        last_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
        change = last_close - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("종목", f"{stock_name} ({ticker})")
        col2.metric("현재가 (원)", f"{last_close:,.0f}", f"{change:+,.0f} ({change_pct:+.2f}%)")
        col3.metric("52주 최고", f"{info.get('fiftyTwoWeekHigh', df['High'].max()):,.0f}")
        col4.metric("52주 최저", f"{info.get('fiftyTwoWeekLow', df['Low'].min()):,.0f}")
        mcap = info.get("marketCap")
        col5.metric("시가총액", f"{mcap / 1e12:,.1f}조" if mcap else "N/A")

        st.divider()
        fig = build_chart(df, stock_name, sma_options, show_rsi, show_macd)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 원본 데이터 보기"):
            st.dataframe(df.sort_index(ascending=False), use_container_width=True)
            csv = df.to_csv().encode("utf-8-sig")
            st.download_button("CSV 다운로드", data=csv, file_name=f"{ticker}_data.csv", mime="text/csv")

        if info.get("longBusinessSummary"):
            with st.expander("🏢 기업 개요"):
                st.write(info["longBusinessSummary"])

# ---------- 탭 2: 종목 비교 ----------
with tab2:
    st.write("여러 종목의 상대 수익률을 같은 시작점(=100)으로 맞춰 비교합니다.")
    default_pick = ["삼성전자", "SK하이닉스", "한미반도체"]
    picked = st.multiselect("비교할 종목 선택 (2개 이상 권장)", list(ALL_STOCKS.keys()), default=default_pick)

    if len(picked) >= 1:
        price_dict = {}
        with st.spinner("데이터를 불러오는 중..."):
            for name in picked:
                code = ALL_STOCKS[name]
                d = load_price_data(code, start_date, end_date)
                if not d.empty:
                    price_dict[name] = d["Close"]

        if price_dict:
            fig_cmp = build_comparison_chart(price_dict)
            st.plotly_chart(fig_cmp, use_container_width=True)

            st.markdown("**기간 수익률 요약**")
            summary = []
            for name, series in price_dict.items():
                ret = (series.iloc[-1] / series.iloc[0] - 1) * 100
                summary.append({"종목": name, "시작가": f"{series.iloc[0]:,.0f}",
                                 "종료가": f"{series.iloc[-1]:,.0f}", "수익률(%)": f"{ret:+.2f}"})
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
        else:
            st.warning("데이터를 불러오지 못했습니다.")
    else:
        st.info("비교할 종목을 하나 이상 선택해주세요.")
