"""투자지표·밸류에이션·유동성 지표 계산 (순수 함수, 금액 단위: 원)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def _div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def _z(v):
    return 0.0 if v is None else v


@dataclass
class Market:
    price: float          # 현재 주가
    shares: float         # 주당 지표 계산에 쓰는 보통주 주식수

    @property
    def mcap(self) -> float:
        return self.price * self.shares


def enterprise_value(flows: dict, stocks: dict, mkt: Market, include_lease: bool) -> dict:
    debt = _z(stocks.get("debt")) + (_z(stocks.get("lease")) if include_lease else 0)
    cash = _z(stocks.get("cash")) + _z(stocks.get("st_fin"))
    net_debt = debt - cash
    nci = _z(stocks.get("equity")) - _z(stocks.get("equity_owner"))  # 비지배지분
    ev = mkt.mcap + net_debt + nci
    ebitda = None if flows.get("op_income") is None else flows["op_income"] + _z(flows.get("da"))
    return {"debt": debt, "cash": cash, "net_debt": net_debt, "nci": nci, "ev": ev, "ebitda": ebitda}


def multiples(flows: dict, stocks: dict, mkt: Market, ev: dict) -> dict:
    ni = flows.get("ni_owner")
    eq = stocks.get("equity_owner")
    rev = flows.get("revenue")
    op = flows.get("op_income")
    eps = _div(ni, mkt.shares)
    bps = _div(eq, mkt.shares)
    sps = _div(rev, mkt.shares)
    fcf = None if flows.get("ocf") is None else flows["ocf"] - abs(_z(flows.get("capex")))
    return {
        "시가총액": mkt.mcap,
        "EPS": eps, "BPS": bps, "SPS": sps,
        "PER": _div(mkt.price, eps) if eps and eps > 0 else None,
        "PBR": _div(mkt.price, bps) if bps and bps > 0 else None,
        "PSR": _div(mkt.price, sps) if sps and sps > 0 else None,
        "EV": ev["ev"], "EBITDA": ev["ebitda"],
        "EV/EBITDA": _div(ev["ev"], ev["ebitda"]) if ev["ebitda"] and ev["ebitda"] > 0 else None,
        "EV/EBIT": _div(ev["ev"], op) if op and op > 0 else None,
        "ROE(%)": None if _div(ni, eq) is None else _div(ni, eq) * 100,
        "영업이익률(%)": None if _div(op, rev) is None else _div(op, rev) * 100,
        "순이익률(%)": None if _div(ni, rev) is None else _div(ni, rev) * 100,
        "FCF": fcf,
        "FCF Yield(%)": None if _div(fcf, mkt.mcap) is None else _div(fcf, mkt.mcap) * 100,
    }


def relative_valuation(m: dict, ev: dict, mkt: Market, targets: dict) -> pd.DataFrame:
    """목표 멀티플 × 주당 지표 → 적정주가."""
    rows = []

    def add(method, metric_nm, metric, target, value):
        rows.append({"방법": method, "기준 지표": metric_nm, "기준값": metric,
                     "적용 멀티플": target, "적정주가": value})

    if targets.get("PER"):
        add("PER", "EPS", m["EPS"], targets["PER"],
            m["EPS"] * targets["PER"] if m["EPS"] and m["EPS"] > 0 else None)
    if targets.get("PBR"):
        add("PBR", "BPS", m["BPS"], targets["PBR"],
            m["BPS"] * targets["PBR"] if m["BPS"] and m["BPS"] > 0 else None)
    if targets.get("PSR"):
        add("PSR", "SPS", m["SPS"], targets["PSR"],
            m["SPS"] * targets["PSR"] if m["SPS"] else None)
    if targets.get("EV/EBITDA"):
        e = ev["ebitda"]
        val = None
        if e and e > 0:
            val = (e * targets["EV/EBITDA"] - ev["net_debt"] - ev["nci"]) / mkt.shares
        add("EV/EBITDA", "EBITDA(억원)", None if e is None else e / 1e8, targets["EV/EBITDA"], val)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["현재가 대비(%)"] = (df["적정주가"] / mkt.price - 1) * 100
    return df


def s_rim(equity_owner: float | None, roe_pct: float | None, k_pct: float,
          shares: float) -> pd.DataFrame:
    """S-RIM: 기업가치 = 자기자본 + 초과이익 / (1 + k - w).

    w = 초과이익 지속계수 (1: 영구지속, 0.9: 매년 10%씩 감소)
    """
    if not equity_owner or roe_pct is None or not k_pct:
        return pd.DataFrame()
    k, roe = k_pct / 100, roe_pct / 100
    excess = equity_owner * (roe - k)
    rows = []
    for w, nm in ((1.0, "초과이익 지속 (w=1.0)"), (0.9, "10%씩 감소 (w=0.9)")):
        value = equity_owner + (excess / k if w == 1.0 else excess * w / (1 + k - w))
        rows.append({"시나리오": nm, "기업가치(원)": value, "적정주가": value / shares})
    return pd.DataFrame(rows)


BAND_PCTS = (0.1, 0.3, 0.5, 0.7, 0.9)


def multiple_band(prices: pd.Series, per_share: pd.Series) -> dict | None:
    """과거 주가 × 주당지표로 멀티플 밴드를 만든다.

    prices: 날짜 index 의 종가 / per_share: '적용 시작일' index 의 주당지표(EPS·BPS 등)
    반환: 시계열 표, 밴드 배수(10·30·50·70·90 백분위), 현재 배수와 그 백분위
    """
    if prices.empty or per_share.dropna().empty:
        return None
    px = prices.rename("price").sort_index().to_frame()
    ps = per_share.rename("metric").sort_index().to_frame()
    df = pd.merge_asof(px, ps, left_index=True, right_index=True).dropna()
    df = df[df["metric"] > 0]
    if len(df) < 10:
        return None
    df["multiple"] = df["price"] / df["metric"]
    levels = df["multiple"].quantile(list(BAND_PCTS)).tolist()
    for p, m in zip(BAND_PCTS, levels):
        df[f"{m:.1f}x"] = df["metric"] * m
    cur = df["multiple"].iloc[-1]
    return {
        "data": df, "levels": levels,
        "current": cur, "median": df["multiple"].median(), "mean": df["multiple"].mean(),
        "percentile": (df["multiple"] < cur).mean() * 100,
    }


# 업종별 '양호' 기준 (제조업 기준에서 바꾸는 항목만)
INDUSTRY_THRESHOLDS = {
    "제조업·기타": {},
    "건설업": {"유동비율": 120, "당좌비율": 80, "부채비율": 200, "자기자본비율": 30, "순차입금비율": 100},
    "유통·서비스업": {"유동비율": 100, "당좌비율": 50, "부채비율": 150, "자기자본비율": 40},
    "금융업": None,  # 예금·보험부채 때문에 유동성 비율 자체가 의미 없음
}


def industry_group(induty_code: str | None) -> str:
    """표준산업분류(KSIC) 코드 앞 2자리로 업종 그룹을 정한다."""
    code = (induty_code or "")[:2]
    if code in ("64", "65", "66"):
        return "금융업"
    if code in ("41", "42"):
        return "건설업"
    if code in ("45", "46", "47", "55", "56"):
        return "유통·서비스업"
    return "제조업·기타"


def debt_ratio_analysis(stocks: dict, prev: dict, since: dict, flows: dict, ev: dict,
                        industry: str) -> dict | None:
    """부채비율(부채/자본) 변화를 원인별로 나누고 규칙 기반으로 해석한다. AI 호출 없음.

    분해: R1 − R0 = ΔL / E1  +  L0 × (1/E1 − 1/E0)
      - 부채 쪽 ΔL 은 차입금·매입채무·선수금·기타로,
      - 자본 쪽 ΔE 는 순이익·배당·자사주·유상증자·기타로 다시 나눈다 (합계가 정확히 맞음).
    """
    L1, E1 = stocks.get("total_liab"), stocks.get("equity")
    L0, E0 = prev.get("total_liab"), prev.get("equity")
    if None in (L1, E1, L0, E0) or E0 <= 0 or E1 <= 0:
        return None
    r0, r1 = L0 / E0 * 100, L1 / E1 * 100

    def dlt(k):
        return _z(stocks.get(k)) - _z(prev.get(k))

    d_debt = dlt("debt") + dlt("lease")
    d_pay, d_adv = dlt("payables"), dlt("advances")
    d_other_l = (L1 - L0) - d_debt - d_pay - d_adv
    ni = _z(since.get("net_income"))
    div = -abs(_z(since.get("dividends_paid")))
    buy = -abs(_z(since.get("buyback")))
    iss = abs(_z(since.get("share_issue")))
    d_other_e = (E1 - E0) - ni - div - buy - iss

    liab_pp = lambda d: d / E1 * 100                 # noqa: E731  부채 증가 → 비율 상승
    eq_pp = lambda d: -L0 * d / (E0 * E1) * 100      # noqa: E731  자본 증가 → 비율 하락
    comps = [
        ("차입금(리스 포함)", liab_pp(d_debt), d_debt),
        ("매입채무", liab_pp(d_pay), d_pay),
        ("선수금·계약부채", liab_pp(d_adv), d_adv),
        ("기타 부채", liab_pp(d_other_l), d_other_l),
        ("순이익", eq_pp(ni), ni),
        ("배당", eq_pp(div), div),
        ("자기주식 취득", eq_pp(buy), buy),
        ("유상증자", eq_pp(iss), iss),
        ("기타 자본변동", eq_pp(d_other_e), d_other_e),
    ]
    comps = [c for c in comps if abs(c[1]) >= 0.05]

    th = (INDUSTRY_THRESHOLDS.get(industry) or {}).get("부채비율", 100)
    delta = r1 - r0
    notes: list[tuple[str, str]] = []   # (수준: good/info/warn/bad, 문장)

    # 1) 수준
    if r1 <= th:
        notes.append(("good", f"부채비율 {r1:.1f}%는 {industry} 기준({th}% 이하) 안에 있습니다."))
    elif r1 <= th * 1.5:
        notes.append(("warn", f"부채비율 {r1:.1f}%로 {industry} 기준({th}%)을 넘었지만 1.5배 이내입니다."))
    else:
        notes.append(("bad", f"부채비율 {r1:.1f}%로 {industry} 기준({th}%)의 1.5배를 넘습니다."))

    # 2) 속도
    if abs(delta) < 5:
        notes.append(("info", f"직전 연말({r0:.1f}%) 대비 {delta:+.1f}%p로 거의 변화가 없습니다."))
    else:
        size = "크게 " if abs(delta) >= 20 else ""
        notes.append(("warn" if delta >= 20 else "info",
                      f"직전 연말({r0:.1f}%) 대비 {delta:+.1f}%p {size}{'상승' if delta > 0 else '하락'}했습니다."))

    # 3) 주요 원인 (영향이 큰 순서로 최대 3개)
    ocf, capex = since.get("ocf"), abs(_z(since.get("capex")))
    for name, pp, amt in sorted(comps, key=lambda c: -abs(c[1]))[:3]:
        if abs(pp) < 1:
            continue
        head = f"{name} {'증가' if amt > 0 else '감소'}({amt / 1e8:+,.0f}억) → {pp:+.1f}%p: "
        if name.startswith("차입금"):
            if amt > 0 and ocf is not None and ocf <= 0:
                notes.append(("bad", head + "영업현금흐름이 마이너스인 상태에서 빚이 늘었습니다. 운영자금 부족 신호일 수 있습니다."))
            elif amt > 0 and ocf and capex >= ocf * 0.8:
                notes.append(("info", head + "같은 기간 설비투자가 영업현금흐름에 맞먹어 투자 목적 차입으로 보입니다."))
            elif amt > 0:
                notes.append(("warn", head + "차입 증가의 용도(투자·운영자금·M&A)를 사업보고서에서 확인할 필요가 있습니다."))
            else:
                notes.append(("good", head + "차입금을 갚아 재무 부담을 줄였습니다(디레버리징)."))
        elif name in ("매입채무", "선수금·계약부채"):
            notes.append(("info", head + ("매출·수주가 늘면서 생기는 영업 관련 부채로, 이자 부담이 없어 차입 증가보다 위험이 낮습니다."
                                          if amt > 0 else "영업 관련 부채가 줄었습니다. 매출·수주 둔화 때문인지 확인해 보세요.")))
        elif name == "순이익":
            notes.append(("good", head + "이익이 쌓여 자본이 늘었습니다.") if amt > 0 else
                         ("bad", head + "순손실로 자본이 줄어 부채비율이 올랐습니다. 가장 주의할 원인입니다."))
        elif name in ("배당", "자기주식 취득"):
            notes.append(("info", head + "주주환원으로 자본이 줄어든 것으로, 회사가 의도한 변화입니다."))
        elif name == "유상증자":
            notes.append(("info", head + "유상증자로 자본을 확충했습니다. 재무는 좋아졌지만 기존 주주 지분은 희석됩니다."))
        elif name == "기타 부채":
            notes.append(("info", head + "미지급금·미지급비용·법인세부채·충당부채 등 이자가 붙지 않는 부채의 변동입니다. "
                                         "실적이 좋아질 때 법인세부채가 함께 늘어나는 경우가 많습니다."
                          if amt > 0 else head + "이자가 붙지 않는 기타 부채가 줄었습니다."))
        else:
            notes.append(("info", head + "금융자산 평가손익, 해외법인 환산차이, 자기주식 소각 등 "
                                         "순이익·배당 밖의 자본 변동입니다."))

    # 4) 교차 확인
    nd = ev.get("net_debt")
    ie, op = flows.get("interest_exp"), flows.get("op_income")
    icr = _div(op, abs(ie)) if ie else None
    if nd is not None and nd < 0:
        notes.append(("good", f"순현금 {abs(nd) / 1e8:,.0f}억 상태라 부채비율과 별개로 실제 상환 위험은 낮습니다."))
    if icr is not None:
        if icr < 1:
            notes.append(("bad", f"이자보상배율 {icr:.1f}배: 영업이익으로 이자도 감당하지 못합니다."))
        elif icr < 3:
            notes.append(("warn", f"이자보상배율 {icr:.1f}배로 낮은 편이라 금리·실적 변화에 민감합니다."))
        elif icr >= 5:
            notes.append(("good", f"이자보상배율 {icr:.1f}배로 이자 부담은 충분히 감당 가능합니다."))

    order = {"good": 0, "info": 1, "warn": 2, "bad": 3}
    worst = max(order[n[0]] for n in notes)
    verdict = ["안정", "보통", "주의", "경고"][worst]
    return {"r0": r0, "r1": r1, "delta": delta, "threshold": th, "components": comps,
            "notes": notes, "verdict": verdict}


# 유동성·안정성: (이름, 값, 단위, 양호 기준, 판단 방향, 설명)
def liquidity(flows: dict, stocks: dict, ev: dict, industry: str = "제조업·기타") -> pd.DataFrame:
    ca, cl = stocks.get("current_assets"), stocks.get("current_liab")
    inv = _z(stocks.get("inventory"))
    cash = ev["cash"]
    eq, tl, ta = stocks.get("equity"), stocks.get("total_liab"), stocks.get("total_assets")
    op, ie = flows.get("op_income"), flows.get("interest_exp")
    pct = lambda x: None if x is None else x * 100  # noqa: E731

    rows = [
        ("유동비율", pct(_div(ca, cl)), "%", 150, "high", "유동자산 ÷ 유동부채 — 1년 내 단기채무 상환능력"),
        ("당좌비율", pct(_div(None if ca is None else ca - inv, cl)), "%", 100, "high",
         "(유동자산 − 재고자산) ÷ 유동부채 — 재고 처분 없이 갚을 수 있는 능력"),
        ("현금비율", pct(_div(cash, cl)), "%", 20, "high", "(현금성자산 + 단기금융상품) ÷ 유동부채"),
        ("부채비율", pct(_div(tl, eq)), "%", 100, "low", "부채총계 ÷ 자본총계"),
        ("자기자본비율", pct(_div(eq, ta)), "%", 50, "high", "자본총계 ÷ 자산총계"),
        ("차입금의존도", pct(_div(ev["debt"], ta)), "%", 30, "low", "총차입금 ÷ 자산총계"),
        ("순차입금비율", pct(_div(ev["net_debt"], eq)), "%", 50, "low",
         "(총차입금 − 현금성자산) ÷ 자본총계, 음수면 순현금"),
        ("이자보상배율", _div(op, abs(ie) if ie else None), "배", 3, "high",
         "영업이익 ÷ 이자비용 — 1배 미만이면 영업으로 이자도 못 갚음"),
        ("Net Debt / EBITDA", _div(ev["net_debt"], ev["ebitda"]) if ev["ebitda"] and ev["ebitda"] > 0 else None,
         "배", 3, "low", "순차입금을 EBITDA로 몇 년이면 갚는지"),
    ]
    overrides = INDUSTRY_THRESHOLDS.get(industry, {})
    out = []
    for name, val, unit, th, direction, desc in rows:
        if overrides is None:
            out.append({"지표": name, "값": val, "단위": unit, "기준": "업종 특성상 기준 없음",
                        "평가": "N/A", "설명": desc})
            continue
        th = overrides.get(name, th)
        if val is None:
            grade = "N/A"
        elif direction == "high":
            grade = "양호" if val >= th else ("보통" if val >= th * 0.67 else "주의")
        else:
            grade = "양호" if val <= th else ("보통" if val <= th * 1.5 else "주의")
        out.append({"지표": name, "값": val, "단위": unit,
                    "기준": f"{'≥' if direction == 'high' else '≤'} {th}{unit}",
                    "평가": grade, "설명": desc})
    return pd.DataFrame(out)
