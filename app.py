"""DART 기업 분석기 (Streamlit).

실행:  streamlit run app.py
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dart_client import REPRT_ANNUAL, DartClient, DartError, Report
from financials import LABELS, Analysis, analyze, annual_series, quarterly_history
from valuation import (INDUSTRY_THRESHOLDS, Market, debt_ratio_analysis, enterprise_value,
                       industry_group, liquidity, multiple_band, multiples, relative_valuation, s_rim)

st.set_page_config(page_title="DART 기업 분석기", page_icon="📈", layout="wide")

COLORS = {"revenue": "#4C78A8", "op_income": "#F58518", "net_income": "#54A24B",
          "margin_op": "#E45756", "margin_ni": "#72B7B2"}
GRADE_COLOR = {"양호": "#2E9E5B", "보통": "#E0A800", "주의": "#D64545", "N/A": "#999999"}


# ------------------------------------------------------------------ 포맷
def eok(v, digits=0):
    if v is None or pd.isna(v):
        return "-"
    if abs(v) >= 1e12:  # 1조 이상은 조 단위로 (지표 카드에서 잘리지 않게)
        return f"{v / 1e12:,.1f}조"
    return f"{v / 1e8:,.{digits}f}억"


def won(v):
    return "-" if v is None or pd.isna(v) else f"{v:,.0f}원"


def times(v):
    return "-" if v is None or pd.isna(v) else f"{v:,.2f}배"


def pct(v):
    return "-" if v is None or pd.isna(v) else f"{v:,.1f}%"


# ------------------------------------------------------------------ 캐시된 API 호출
@st.cache_resource
def get_client(api_key: str) -> DartClient:
    return DartClient(api_key)


@st.cache_data(ttl=86400, show_spinner="회사 목록을 불러오는 중… (최초 1회)")
def cached_search(api_key: str, kw: str) -> pd.DataFrame:
    return get_client(api_key).search_corp(kw)


@st.cache_data(ttl=3600, show_spinner="정기보고서 목록 조회 중…")
def cached_reports(api_key: str, corp_code: str) -> list[Report]:
    return get_client(api_key).periodic_reports(corp_code)


@st.cache_data(ttl=3600, show_spinner="재무제표 분석 중…")
def cached_analysis(api_key: str, corp_code: str, corp_name: str, rep: Report, fs_div: str) -> Analysis:
    return analyze(get_client(api_key), corp_code, corp_name, rep, fs_div)


@st.cache_data(ttl=3600)
def cached_shares(api_key: str, corp_code: str, rep: Report) -> dict:
    c = get_client(api_key)
    out = c.share_count(corp_code, rep.bsns_year, rep.reprt_code)
    if out["issued"] is None and not rep.is_annual:
        out = c.share_count(corp_code, str(int(rep.bsns_year) - 1), REPRT_ANNUAL)
    return out


@st.cache_data(ttl=86400)
def cached_induty_code(api_key: str, corp_code: str) -> str:
    try:
        return get_client(api_key).company(corp_code).get("induty_code", "")
    except Exception:  # 업종을 몰라도 분석은 계속 (제조업 기준 적용)
        return ""


@st.cache_data(ttl=3600, show_spinner="분기 실적 조회 중…")
def cached_quarterly(api_key: str, corp_code: str, rep: Report, fs_div: str) -> pd.DataFrame:
    return quarterly_history(get_client(api_key), corp_code, rep, fs_div)


@st.cache_data(ttl=3600, show_spinner="과거 재무 조회 중…")
def cached_annual_series(api_key: str, corp_code: str, last_year: int, fs_div: str) -> pd.DataFrame:
    return annual_series(get_client(api_key), corp_code, last_year, fs_div)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_corp_list(api_key: str) -> pd.DataFrame:
    return get_client(api_key).corp_codes()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_price_history(stock_code: str, years: int = 6) -> pd.Series:
    try:
        import FinanceDataReader as fdr
        df = fdr.DataReader(stock_code, (date.today() - timedelta(days=365 * years)).isoformat())
        return df["Close"].resample("W-FRI").last().dropna()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_peer(api_key: str, corp_code: str, corp_name: str, stock_code: str, fs_div: str) -> dict:
    """동종업계 비교용: 피어의 가장 최근 정기보고서 기준 멀티플."""
    row = {"회사": f"{corp_name} ({stock_code})"}
    try:
        reps = cached_reports(api_key, corp_code)
        if not reps:
            return {**row, "비고": "정기보고서 없음"}
        p_rep = reps[0]
        pa = cached_analysis(api_key, corp_code, corp_name, p_rep, fs_div)
        sh = cached_shares(api_key, corp_code, p_rep)
        px, _ = cached_price(stock_code)
        n = sh["floating"] or sh["issued"]
        if not px or not n:
            return {**row, "기준": pa.flows_basis, "비고": "주가·주식수 조회 실패"}
        p_mkt = Market(px, n)
        p_ev = enterprise_value(pa.flows, pa.stocks, p_mkt, include_lease=False)
        pm = multiples(pa.flows, pa.stocks, p_mkt, p_ev)
        return {**row, "기준": p_rep.period_end, **_peer_cols(pm, pa.stocks), "비고": ""}
    except Exception as e:  # 피어 하나가 실패해도 나머지는 보여준다
        return {**row, "비고": f"오류: {e}"[:60]}


def _peer_cols(m: dict, stocks: dict) -> dict:
    tl, eq = stocks.get("total_liab"), stocks.get("equity")
    return {"시가총액": m["시가총액"], "PER": m["PER"], "PBR": m["PBR"], "PSR": m["PSR"],
            "EV/EBITDA": m["EV/EBITDA"], "ROE(%)": m["ROE(%)"], "영업이익률(%)": m["영업이익률(%)"],
            "부채비율(%)": tl / eq * 100 if tl is not None and eq else None}


@st.cache_data(ttl=600)
def cached_price(stock_code: str) -> tuple[float | None, str]:
    try:
        import FinanceDataReader as fdr
        df = fdr.DataReader(stock_code, (date.today() - timedelta(days=14)).isoformat())
        if not df.empty:
            return float(df["Close"].iloc[-1]), df.index[-1].strftime("%Y-%m-%d")
    except Exception:  # 네트워크·라이브러리 오류 시 수동 입력으로 대체
        pass
    return None, ""


def server_api_key() -> str:
    """서버에 저장된 키 (배포 시 Secrets, 로컬은 환경변수). 있으면 화면에 입력칸을 아예 보여주지 않는다."""
    try:
        if "DART_API_KEY" in st.secrets:
            return st.secrets["DART_API_KEY"]
    except Exception:  # secrets.toml 이 없으면 예외
        pass
    return os.getenv("DART_API_KEY", "")


# ------------------------------------------------------------------ 사이드바
with st.sidebar:
    st.header("📈 DART 기업 분석기")
    api_key = server_api_key()
    if not api_key:
        api_key = st.text_input("OpenDART API 키", type="password",
                                help="opendart.fss.or.kr 에서 무료 발급. 서버 Secrets 나 환경변수 DART_API_KEY 로 "
                                     "설정해 두면 이 칸은 나타나지 않습니다")
    kw = st.text_input("회사명 또는 종목코드", placeholder="예: 삼성전자, 005930")

    corp = None
    rep = None
    fs_div = "CFS"
    if api_key and kw:
        try:
            hits = cached_search(api_key, kw)
        except (DartError, Exception) as e:
            st.error(f"회사 검색 실패: {e}")
            hits = pd.DataFrame()
        if hits.empty:
            st.info("검색 결과가 없습니다. (상장사만 검색됩니다)")
        else:
            idx = st.selectbox("회사 선택", range(len(hits)),
                               format_func=lambda i: f"{hits.iloc[i].corp_name} ({hits.iloc[i].stock_code})")
            corp = hits.iloc[idx]
            try:
                reports = cached_reports(api_key, corp.corp_code)
            except DartError as e:
                st.error(str(e))
                reports = []
            if reports:
                ri = st.selectbox("보고서 선택", range(len(reports)),
                                  format_func=lambda i: reports[i].label)
                rep = reports[ri]
            else:
                st.warning("최근 정기보고서가 없습니다.")
            fs_div = "CFS" if st.radio("재무제표", ["연결", "별도"], horizontal=True) == "연결" else "OFS"

    run = st.button("분석 실행", type="primary", width="stretch", disabled=rep is None)
    if run:
        st.session_state["job"] = (api_key, corp.corp_code, corp.corp_name, corp.stock_code, rep, fs_div)
    st.caption("데이터: 금융감독원 OpenDART · 주가: FinanceDataReader")

job = st.session_state.get("job")
if not job:
    st.title("DART 기업 분석기")
    st.markdown(
        """
1. 왼쪽 사이드바에서 회사명이나 종목코드로 검색하세요.
2. 회사를 고르고 **사업보고서 / 반기 / 분기보고서** 중 하나를 고른 뒤 **분석 실행**을 누르세요.

제공 기능
- 최근 3개년 매출액·영업이익·당기순이익 추이와 이익률 (분기보고서 선택 시 TTM 포함)
- PER · PBR · PSR · EV/EBITDA · EV/EBIT · ROE · FCF Yield
- 목표 멀티플 기반 적정주가, S-RIM, 적정주가 밴드 차트
- 유동비율 · 당좌비율 · 현금비율 · 부채비율 · 이자보상배율 등 유동성·안정성 진단
"""
    )
    st.stop()

api_key, corp_code, corp_name, stock_code, rep, fs_div = job
try:
    A = cached_analysis(api_key, corp_code, corp_name, rep, fs_div)
except (DartError, ValueError) as e:
    st.error(str(e))
    st.stop()

# ------------------------------------------------------------------ 헤더 & 시장 데이터
st.title(f"{A.corp_name} ({stock_code})")
st.caption(f"[{rep.report_nm.strip()}]({rep.url}) · {'연결' if A.fs_div == 'CFS' else '별도'}재무제표 · "
           f"손익 기준: **{A.flows_basis}** · 재무상태 기준: {rep.period_end} 말")
for w in A.warnings:
    st.warning(w, icon="⚠️")

shares_info = cached_shares(api_key, corp_code, rep)
fetched_price, price_date = cached_price(stock_code)

c1, c2 = st.columns([1, 3])
price = c1.number_input(f"주가 (원){' · ' + price_date if price_date else ''}",
                        value=float(fetched_price or 10000), step=100.0, min_value=1.0)
if not fetched_price:
    c1.caption("⚠️ 주가 자동조회 실패 — 직접 입력하세요")

# 감가상각비를 못 찾았으면 직접 입력하도록 펼쳐서 보여준다
with c2.expander("⚙️ 고급 설정 (주식수 · 리스부채 · 감가상각비)", expanded=A.flows.get("da") is None):
    a1, a2, a3 = st.columns(3)
    share_mode = a1.radio("주식수 기준", ["유통주식수(자기주식 제외)", "발행주식총수"])
    default_shares = shares_info["floating"] if share_mode.startswith("유통") else shares_info["issued"]
    shares = a2.number_input("보통주 주식수", value=float(default_shares or 1_000_000), step=1000.0,
                             min_value=1.0, format="%.0f")
    include_lease = a3.checkbox("리스부채를 차입금에 포함", value=False,
                                help="EV·순차입금 계산 시 리스부채(IFRS16)를 부채로 볼지 여부")
    da_override = a3.number_input("D&A 직접입력 (억원, 0=자동)", value=0.0, step=10.0,
                                  help="사업보고서 주석의 감가상각비·무형자산상각비 합계")

flows = dict(A.flows)
if da_override:
    flows["da"] = da_override * 1e8
mkt = Market(price=price, shares=shares)
EV = enterprise_value(flows, A.stocks, mkt, include_lease)
M = multiples(flows, A.stocks, mkt, EV)

k = st.columns(7)
k[0].metric("시가총액", eok(M["시가총액"]))
k[1].metric("PER", times(M["PER"]))
k[2].metric("PBR", times(M["PBR"]))
k[3].metric("PSR", times(M["PSR"]))
k[4].metric("EV/EBITDA", times(M["EV/EBITDA"]))
k[5].metric("ROE", pct(M["ROE(%)"]))
k[6].metric("순차입금", eok(EV["net_debt"]), help="음수면 순현금")

# ------------------------------------------------------------------ 밴드·피어 (목표 멀티플 채우기에 쓰므로 탭보다 먼저 계산)
annual_last = int(rep.bsns_year) if rep.is_annual else int(rep.bsns_year) - 1
series = cached_annual_series(api_key, corp_code, annual_last, A.fs_div)
px_hist = cached_price_history(stock_code)


def _per_share(col: str, latest_val) -> pd.Series:
    """연간 값은 다음 해 4월 1일(사업보고서 공시 후)부터, 최신 TTM 값은 보고서 접수일부터 적용."""
    s = pd.Series({pd.Timestamp(fy + 1, 4, 1): v for fy, v in series[col].items()}) if not series.empty \
        else pd.Series(dtype=float)
    if latest_val is not None:
        s[pd.Timestamp(rep.rcept_dt)] = latest_val
    return s.sort_index() / shares


bands = {
    "PER": multiple_band(px_hist, _per_share("ni_owner", flows.get("ni_owner"))),
    "PBR": multiple_band(px_hist, _per_share("equity_owner", A.stocks.get("equity_owner"))),
}

peer_key = f"peers_{corp_code}"
peer_rows = []
for opt in st.session_state.get(peer_key, []):
    p_name, p_stock = opt.rsplit(" (", 1)
    p_stock = p_stock.rstrip(")")
    corp_list = cached_corp_list(api_key)
    p_code = corp_list.loc[corp_list["stock_code"] == p_stock, "corp_code"]
    if not p_code.empty:
        with st.spinner(f"{p_name} 분석 중…"):
            peer_rows.append(cached_peer(api_key, p_code.iloc[0], p_name, p_stock, fs_div))
peer_df = pd.DataFrame(peer_rows)
peer_median = {k: pd.to_numeric(peer_df[k], errors="coerce").median() for k in ("PER", "PBR", "PSR", "EV/EBITDA")
               if k in peer_df} if not peer_df.empty else {}


def fill_targets(values: dict):
    for name, v in values.items():
        if v is not None and not pd.isna(v):
            st.session_state[f"tgt_{name}_{corp_code}"] = round(float(v), 2)


tabs = st.tabs(["📊 실적 추이", "💰 밸류에이션", "📈 PER·PBR 밴드", "🏢 동종업계 비교", "💧 유동성·안정성",
                "🎯 적정주가 종합", "📄 재무 데이터"])

# ------------------------------------------------------------------ 1. 실적 추이
with tabs[0]:
    hist = A.history.copy()
    if not rep.is_annual:
        hist.loc["TTM"] = pd.Series({k_: flows.get(k_) for k_ in hist.columns})
    if hist.empty:
        st.info("연간 실적 데이터를 불러오지 못했습니다.")
    else:
        h = hist.astype(float) / 1e8
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for key in ("revenue", "op_income", "net_income"):
            fig.add_bar(x=h.index, y=h[key], name=LABELS[key], marker_color=COLORS[key],
                        text=h[key].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}"),
                        textposition="outside", secondary_y=False)
        fig.add_scatter(x=h.index, y=h["op_income"] / h["revenue"] * 100, name="영업이익률(%)",
                        mode="lines+markers", line=dict(color=COLORS["margin_op"], width=3), secondary_y=True)
        fig.add_scatter(x=h.index, y=h["net_income"] / h["revenue"] * 100, name="순이익률(%)",
                        mode="lines+markers", line=dict(color=COLORS["margin_ni"], width=3, dash="dot"),
                        secondary_y=True)
        fig.update_layout(barmode="group", height=500,
                          legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
                          margin=dict(t=50, b=40), title="매출액 · 영업이익 · 당기순이익 (단위: 억원)")
        fig.update_yaxes(title_text="억원", tickformat=",", secondary_y=False)
        fig.update_yaxes(title_text="이익률(%)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, width="stretch")

        tbl = h.rename(columns=LABELS).T
        growth = h[["revenue", "op_income", "net_income"]].pct_change(fill_method=None) * 100
        st.markdown("**연간 실적 (억원)**")
        st.dataframe(tbl.style.format("{:,.0f}", na_rep="-"), width="stretch")
        st.markdown("**전년 대비 증가율 (%)**")
        st.dataframe(growth.rename(columns=LABELS).T.iloc[:, 1:].style.format("{:+.1f}%", na_rep="-"),
                     width="stretch")

    if A.ytd is not None:
        st.markdown(f"**{rep.report_nm.strip()} 누적 실적 vs 전년 동기 (억원)**")
        y = A.ytd.copy()
        y[["당기 누적", "전년 동기 누적"]] = y[["당기 누적", "전년 동기 누적"]] / 1e8
        st.dataframe(y.style.format({"당기 누적": "{:,.0f}", "전년 동기 누적": "{:,.0f}",
                                     "YoY(%)": "{:+.1f}%"}, na_rep="-"), width="stretch")

    st.divider()
    st.subheader("분기 실적 추이")
    qh = cached_quarterly(api_key, corp_code, rep, A.fs_div)
    if qh.empty:
        st.info("분기 실적을 불러오지 못했습니다.")
    else:
        q = qh / 1e8
        qfig = make_subplots(specs=[[{"secondary_y": True}]])
        qfig.add_bar(x=q.index, y=q["revenue"], name="매출액", marker_color=COLORS["revenue"], secondary_y=False)
        qfig.add_bar(x=q.index, y=q["op_income"], name="영업이익", marker_color=COLORS["op_income"],
                     secondary_y=False)
        qfig.add_scatter(x=q.index, y=q["op_income"] / q["revenue"] * 100, name="영업이익률(%)",
                         mode="lines+markers", line=dict(color=COLORS["margin_op"], width=3), secondary_y=True)
        qfig.update_layout(barmode="group", height=420, title="분기별 매출액 · 영업이익 (단독 분기, 억원)",
                           legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
                           margin=dict(t=50, b=40))
        qfig.update_yaxes(tickformat=",", secondary_y=False)
        qfig.update_yaxes(title_text="이익률(%)", secondary_y=True, showgrid=False)
        st.plotly_chart(qfig, width="stretch")

        qt = q.rename(columns=LABELS)
        yoy = (qh - qh.shift(4)) / qh.shift(4).abs() * 100  # 전년 같은 분기 대비
        qt["매출 YoY(%)"] = yoy["revenue"]
        qt["영업이익 YoY(%)"] = yoy["op_income"]
        qt["영업이익률(%)"] = qh["op_income"] / qh["revenue"] * 100
        st.dataframe(qt.T.style.format(lambda v: "-" if pd.isna(v) else f"{v:,.1f}"), width="stretch")
        st.caption("1~3분기는 각 보고서의 3개월 금액, 4분기는 연간 − 3분기 누적으로 계산합니다. YoY는 전년 같은 분기 대비.")

# ------------------------------------------------------------------ 2. 밸류에이션
with tabs[1]:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("현재 투자지표")
        rows = [
            ("EPS (지배주주)", won(M["EPS"])), ("BPS (지배주주)", won(M["BPS"])), ("SPS", won(M["SPS"])),
            ("PER", times(M["PER"])), ("PBR", times(M["PBR"])), ("PSR", times(M["PSR"])),
            ("EV", eok(EV["ev"])), ("EBITDA", eok(EV["ebitda"])),
            ("EV/EBITDA", times(M["EV/EBITDA"])), ("EV/EBIT", times(M["EV/EBIT"])),
            ("ROE", pct(M["ROE(%)"])),
            ("영업이익률", pct(M["영업이익률(%)"])), ("순이익률", pct(M["순이익률(%)"])),
            ("FCF (영업CF − CAPEX)", eok(M["FCF"])), ("FCF Yield", pct(M["FCF Yield(%)"])),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["지표", "값"]), hide_index=True, width="stretch",
                     height=560)
        with st.expander("EV 구성"):
            st.write(pd.DataFrame({
                "항목": ["시가총액", "+ 총차입금", "− 현금·단기금융상품", "+ 비지배지분", "= EV"],
                "금액": [eok(mkt.mcap), eok(EV["debt"]), eok(EV["cash"]), eok(EV["nci"]), eok(EV["ev"])],
            }))
    with right:
        st.subheader("상대가치: 목표 멀티플 적용")
        st.caption("동종업계 평균이나 이 회사의 과거 평균 멀티플을 넣으세요. 입력한 항목만 계산합니다.")
        b1, b2 = st.columns(2)
        b1.button("동종업계 중앙값으로 채우기", on_click=fill_targets, args=(peer_median,),
                  disabled=not peer_median, width="stretch",
                  help="'동종업계 비교' 탭에서 비교 기업을 고르면 사용할 수 있습니다")
        hist_median = {k: b["median"] for k, b in bands.items() if b}
        b2.button("과거 PER·PBR 중앙값으로 채우기", on_click=fill_targets, args=(hist_median,),
                  disabled=not hist_median, width="stretch")
        t = st.columns(4)
        targets = {}
        for col, (name, step, ph) in zip(t, [("PER", 0.5, "예: 12"), ("PBR", 0.1, "예: 1.2"),
                                             ("PSR", 0.1, "예: 1.5"), ("EV/EBITDA", 0.5, "예: 7")]):
            key = f"tgt_{name}_{corp_code}"
            st.session_state.setdefault(key, None)
            targets[name] = col.number_input(f"목표 {name}", step=step, placeholder=ph, key=key)
        rel = relative_valuation(M, EV, mkt, targets)
        if rel.empty:
            st.info("목표 멀티플을 하나 이상 입력하면 적정주가가 계산됩니다.")
        else:
            st.dataframe(rel.style.format({"기준값": "{:,.0f}", "적용 멀티플": "{:.1f}x", "적정주가": "{:,.0f}원",
                                           "현재가 대비(%)": "{:+.1f}%"}, na_rep="-"),
                         hide_index=True, width="stretch")

        st.subheader("절대가치: S-RIM (잔여이익모델)")
        rc = st.columns(2)
        k_req = rc[0].number_input("요구수익률 k (%)", value=8.0, step=0.25,
                                   help="보통 BBB- 회사채 5년 수익률을 사용")
        roe_in = rc[1].number_input("적용 ROE (%)", value=float(round(M["ROE(%)"] or 0, 2)), step=0.5,
                                    help="기본값은 TTM/연간 ROE. 컨센서스·과거 평균으로 바꿔도 됩니다")
        srim = s_rim(A.stocks.get("equity_owner"), roe_in, k_req, shares)
        if srim.empty:
            st.info("자기자본 또는 ROE가 없어 계산할 수 없습니다.")
        else:
            srim["현재가 대비(%)"] = (srim["적정주가"] / price - 1) * 100
            st.dataframe(srim.style.format({"기업가치(원)": lambda v: eok(v), "적정주가": "{:,.0f}원",
                                            "현재가 대비(%)": "{:+.1f}%"}),
                         hide_index=True, width="stretch")
            if roe_in < k_req:
                st.caption("ROE < k 인 경우 초과이익이 음수 → 감소 시나리오(w<1)가 오히려 높게 나옵니다.")

# ------------------------------------------------------------------ 3. PER·PBR 밴드
with tabs[2]:
    st.subheader("과거 멀티플 밴드")
    st.caption("주가와 그 시점의 주당순이익(EPS)·주당순자산(BPS)으로 과거 PER·PBR을 구하고, "
               "그 분포의 10·30·50·70·90% 지점을 선으로 그립니다. 연간 실적은 다음 해 4월부터, "
               "최신 TTM 실적은 보고서 접수일부터 적용합니다. 주당 값은 현재 주식수 기준입니다.")
    band_colors = ["#2E9E5B", "#8BC34A", "#9E9E9E", "#FF9800", "#D64545"]
    for name, b in bands.items():
        if not b:
            st.info(f"{name} 밴드를 그릴 데이터가 부족합니다. (적자 기간이 길거나 주가 이력 조회 실패)")
            continue
        d = b["data"]
        bfig = go.Figure()
        for (lvl, color) in zip(b["levels"], band_colors):
            bfig.add_scatter(x=d.index, y=d[f"{lvl:.1f}x"], name=f"{name} {lvl:.1f}배",
                             mode="lines", line=dict(color=color, width=1.5, dash="dot"))
        bfig.add_scatter(x=d.index, y=d["price"], name="주가", mode="lines", line=dict(color="#222", width=2.5))
        bfig.update_layout(height=420, title=f"{name} 밴드", margin=dict(t=50, b=20),
                           legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5))
        bfig.update_yaxes(tickformat=",", title_text="원")
        st.plotly_chart(bfig, width="stretch")
        c_1, c_2, c_3, c_4 = st.columns(4)
        c_1.metric(f"현재 {name}", times(b["current"]))
        c_2.metric("과거 중앙값", times(b["median"]))
        c_3.metric("과거 평균", times(b["mean"]))
        c_4.metric("현재 위치", f"하위 {b['percentile']:.0f}%",
                   help="과거 기간 중 지금보다 배수가 낮았던 비율. 낮을수록 역사적으로 싼 구간")

# ------------------------------------------------------------------ 4. 동종업계 비교
with tabs[3]:
    st.subheader("동종업계 비교")
    corp_list = cached_corp_list(api_key)
    options = (corp_list["corp_name"] + " (" + corp_list["stock_code"] + ")").tolist()
    st.multiselect("비교할 기업 (최대 8개)", options, max_selections=8, key=peer_key,
                   placeholder="회사명을 입력해 검색", help="각 기업의 가장 최근 정기보고서 기준으로 계산합니다")
    if peer_df.empty:
        st.info("비교할 기업을 고르면 PER·PBR·ROE 등을 나란히 보여주고, 중앙값을 목표 멀티플로 쓸 수 있습니다.")
    else:
        me = {"회사": f"▶ {A.corp_name} ({stock_code})", "기준": rep.period_end, **_peer_cols(M, A.stocks),
              "비고": ""}
        metric_cols = ("PER", "PBR", "PSR", "EV/EBITDA", "ROE(%)", "영업이익률(%)", "부채비율(%)")
        med = {"회사": "동종업계 중앙값", "기준": "",
               **{c: pd.to_numeric(peer_df[c], errors="coerce").median()
                  for c in ("시가총액", *metric_cols) if c in peer_df},
               "비고": ""}
        table = pd.DataFrame([me, *peer_rows, med]).set_index("회사")
        num_cols = [c for c in metric_cols if c in table]
        table[["시가총액", *num_cols]] = table[["시가총액", *num_cols]].apply(pd.to_numeric, errors="coerce")
        st.dataframe(
            table.style.format({"시가총액": lambda v: eok(v), **{c: "{:.2f}" for c in num_cols}}, na_rep="-"),
            width="stretch")
        pick = st.radio("차트 지표", num_cols, horizontal=True)
        chart = table[pick].drop("동종업계 중앙값").dropna()
        pfig = go.Figure(go.Bar(
            x=chart.values, y=chart.index, orientation="h",
            marker_color=["#E45756" if i.startswith("▶") else "#4C78A8" for i in chart.index],
            text=[f"{v:.2f}" for v in chart.values], textposition="outside"))
        if not pd.isna(table.loc["동종업계 중앙값", pick]):
            pfig.add_vline(x=table.loc["동종업계 중앙값", pick], line_dash="dash",
                           annotation_text="중앙값", annotation_position="top")
        pfig.update_layout(height=80 + 40 * len(chart), margin=dict(t=30, b=20))
        pfig.update_yaxes(autorange="reversed")
        st.plotly_chart(pfig, width="stretch")
        st.caption("적자 기업의 PER, EBITDA가 음수인 기업의 EV/EBITDA는 계산하지 않아 '-'로 표시되고 중앙값에서 빠집니다.")

# ------------------------------------------------------------------ 5. 유동성
with tabs[4]:
    st.subheader("유동성 · 재무안정성")
    detected = industry_group(cached_induty_code(api_key, corp_code))
    groups = list(INDUSTRY_THRESHOLDS)
    industry = st.selectbox("평가 기준 업종", groups, index=groups.index(detected),
                            help="DART 기업개황의 표준산업분류로 자동 선택됩니다. 맞지 않으면 바꾸세요.",
                            key=f"industry_{corp_code}")
    if industry == "금융업":
        st.warning("금융업은 예금·보험부채가 부채 대부분이라 유동비율·부채비율 같은 일반 기준이 맞지 않습니다. "
                   "값은 참고로만 보여주고 평가는 하지 않습니다. (BIS 비율·지급여력비율 등은 DART 재무제표 API에 없음)",
                   icon="⚠️")
    L = liquidity(flows, A.stocks, EV, industry)
    cols = st.columns(3)
    for i, r in L.iterrows():
        with cols[i % 3]:
            v = "-" if r["값"] is None or pd.isna(r["값"]) else f"{r['값']:,.1f}{r['단위']}"
            st.markdown(
                f"""<div style="border:1px solid #ddd;border-radius:10px;padding:12px 14px;margin-bottom:12px">
<div style="display:flex;justify-content:space-between;align-items:center">
<b>{r['지표']}</b><span style="background:{GRADE_COLOR[r['평가']]};color:white;border-radius:6px;
padding:1px 8px;font-size:0.8em">{r['평가']}</span></div>
<div style="font-size:1.7em;font-weight:700;margin:4px 0">{v}</div>
<div style="color:#888;font-size:0.8em">기준 {r['기준']} · {r['설명']}</div></div>""",
                unsafe_allow_html=True)
    st.caption(f"평가 기준: {industry} 참고치. 같은 업종 안에서도 사업 구조에 따라 적정 수준이 다를 수 있습니다.")

    st.divider()
    st.subheader("부채비율 변화 해석")
    D = None if industry == "금융업" else \
        debt_ratio_analysis(A.stocks, A.stocks_prev, A.since_flows, flows, EV, industry)
    if industry == "금융업":
        st.info("금융업은 부채비율 해석을 제공하지 않습니다.")
    elif D is None:
        st.info("직전 연말 재무상태가 없어 변화를 분해할 수 없습니다.")
    else:
        st.markdown(f"**종합: {D['verdict']}** · 직전 연말 {D['r0']:.1f}% → 현재 {D['r1']:.1f}% "
                    f"({D['delta']:+.1f}%p) · 비교 기간: 직전 연말 → {rep.period_end}")
        icon = {"good": "✅", "info": "ℹ️", "warn": "⚠️", "bad": "🚨"}
        for level, text in D["notes"]:
            st.markdown(f"{icon[level]} {text}")

        dc = st.columns(2)
        with dc[0]:
            comps = D["components"]
            wf = go.Figure(go.Waterfall(
                orientation="v",
                measure=["absolute"] + ["relative"] * len(comps) + ["total"],
                x=["직전 연말"] + [c[0] for c in comps] + ["현재"],
                y=[D["r0"]] + [c[1] for c in comps] + [0],
                text=[f"{D['r0']:.1f}%"] + [f"{c[1]:+.1f}%p" for c in comps] + [f"{D['r1']:.1f}%"],
                increasing={"marker": {"color": "#D64545"}}, decreasing={"marker": {"color": "#2E9E5B"}},
                totals={"marker": {"color": "#4C78A8"}}, connector={"line": {"color": "#bbb"}},
            ))
            wf.add_hline(y=D["threshold"], line_dash="dash", line_color="#888",
                         annotation_text=f"{industry} 기준 {D['threshold']}%", annotation_position="top left")
            wf.update_layout(height=420, title="부채비율 변화 원인 (%p)", margin=dict(t=50, b=20))
            st.plotly_chart(wf, width="stretch")
            st.caption("빨강 = 부채비율을 올린 요인, 초록 = 내린 요인. 부채 쪽은 증가분 ÷ 현재 자본, "
                       "자본 쪽은 자본 변동이 비율에 준 영향으로 나눠 합계가 정확히 맞습니다.")
        with dc[1]:
            trend = pd.Series(dtype=float)
            if not series.empty:
                trend = (series["total_liab"] / series["equity"] * 100).dropna()
                trend.index = [f"FY{y}" for y in trend.index]
            if not rep.is_annual:
                trend[rep.period_end] = D["r1"]
            tfig = go.Figure(go.Scatter(x=trend.index, y=trend.values, mode="lines+markers+text",
                                        text=[f"{v:.0f}%" for v in trend.values], textposition="top center",
                                        line=dict(color="#4C78A8", width=3)))
            tfig.add_hline(y=D["threshold"], line_dash="dash", line_color="#888",
                           annotation_text=f"기준 {D['threshold']}%", annotation_position="top left")
            if not peer_df.empty and "부채비율(%)" in peer_df:
                pm = pd.to_numeric(peer_df["부채비율(%)"], errors="coerce").median()
                if not pd.isna(pm):
                    tfig.add_hline(y=pm, line_dash="dot", line_color="#F58518",
                                   annotation_text=f"동종업계 중앙값 {pm:.0f}%", annotation_position="bottom left")
            tfig.update_layout(height=420, title="부채비율 추이", margin=dict(t=50, b=20))
            tfig.update_yaxes(ticksuffix="%")
            st.plotly_chart(tfig, width="stretch")
        st.caption("해석 문장은 정해진 규칙으로 만들어지며 AI를 호출하지 않습니다 (추가 비용 없음).")

    s = A.stocks
    bs = pd.DataFrame({"금액(억원)": {
        "유동자산": s.get("current_assets"), "  재고자산": s.get("inventory"),
        "  현금및현금성자산": s.get("cash"), "  단기금융상품": s.get("st_fin"),
        "유동부채": s.get("current_liab"), "총차입금": s.get("debt"), "리스부채": s.get("lease"),
        "부채총계": s.get("total_liab"), "자본총계": s.get("equity"), "자산총계": s.get("total_assets"),
    }}).astype(float) / 1e8
    st.dataframe(bs.style.format("{:,.0f}", na_rep="-"), width="stretch")

# ------------------------------------------------------------------ 6. 종합
with tabs[5]:
    st.subheader("적정주가 밴드 (Football Field)")
    vals = []
    for _, r in rel.iterrows():
        if r["적정주가"] is not None and not pd.isna(r["적정주가"]):
            vals.append((f"{r['방법']} {r['적용 멀티플']:.1f}x", r["적정주가"]))
    for _, r in srim.iterrows():
        vals.append((f"S-RIM {r['시나리오'].split(' ')[-1]}", r["적정주가"]))
    if not vals:
        st.info("계산된 적정주가가 없습니다.")
    else:
        df_v = pd.DataFrame(vals, columns=["방법", "적정주가"])
        fig = go.Figure(go.Bar(
            y=df_v["방법"], x=df_v["적정주가"], orientation="h",
            marker_color=["#2E9E5B" if v >= price else "#D64545" for v in df_v["적정주가"]],
            text=df_v["적정주가"].map(lambda v: f"{v:,.0f}원"), textposition="outside",
        ))
        fig.add_vline(x=price, line_dash="dash", line_color="#333",
                      annotation_text=f"현재가 {price:,.0f}원", annotation_position="top")
        fig.update_layout(height=100 + 45 * len(df_v), margin=dict(t=40, b=20), xaxis_title="원")
        fig.update_xaxes(tickformat=",")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")
        # 방법마다 전제가 달라 평균은 내지 않고 범위와 현재가 위치만 보여준다
        lo, hi = df_v["적정주가"].min(), df_v["적정주가"].max()
        above = int((df_v["적정주가"] >= price).sum())
        m1, m2 = st.columns(2)
        m1.metric("적정주가 범위", f"{lo:,.0f} ~ {hi:,.0f}원")
        m2.metric("현재가보다 높게 나온 방법", f"{above} / {len(df_v)}개")
    st.caption("※ 본 결과는 공시 재무데이터와 입력 가정에 기반한 참고용 계산이며 투자 권유가 아닙니다.")

# ------------------------------------------------------------------ 7. 재무 데이터
with tabs[6]:
    st.subheader("분석에 사용된 계정")
    used = pd.DataFrame(
        [(LABELS[k_], "손익·현금흐름 (" + A.flows_basis + ")", v) for k_, v in flows.items()] +
        [(LABELS[k_], f"재무상태 ({rep.period_end})", v) for k_, v in A.stocks.items()],
        columns=["계정", "구분", "금액(원)"],
    )
    used["금액(억원)"] = used["금액(원)"].astype(float) / 1e8
    st.dataframe(used.style.format({"금액(원)": "{:,.0f}", "금액(억원)": "{:,.1f}"}, na_rep="미발견"),
                 hide_index=True, width="stretch")
    st.caption("계정명은 회사마다 달라 자동 매칭됩니다. '미발견' 항목이나 이상한 값은 아래 원본에서 확인하세요.")
    with st.expander("DART 원본 재무제표 (fnlttSinglAcntAll)"):
        raw_cols = [c for c in ["sj_nm", "account_nm", "account_id", "thstrm_nm", "thstrm_amount",
                                "thstrm_add_amount", "frmtrm_nm", "frmtrm_amount", "frmtrm_add_amount",
                                "bfefrmtrm_amount"] if c in A.raw.columns]
        st.dataframe(A.raw[raw_cols], width="stretch", height=500)
        st.download_button("CSV 다운로드", A.raw.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{A.corp_name}_{rep.bsns_year}_{rep.reprt_code}.csv", mime="text/csv")
