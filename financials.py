"""DART 전체 재무제표(fnlttSinglAcntAll)에서 분석에 필요한 계정을 뽑아낸다.

회사마다 계정명이 조금씩 달라서 1) 표준 account_id 2) 계정명 정규식 순으로 찾는다.
금액 단위는 모두 '원'.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from dart_client import (REPRT_ANNUAL, REPRT_HALF, REPRT_Q1, REPRT_Q3, DartClient, Report,
                         to_num)

IS_ = ("IS", "CIS")
BS_ = ("BS",)
CF_ = ("CF",)

# key: (재무제표 구분, account_id 후보, 계정명 정규식 후보, 제외 정규식, 합산 여부)
ITEMS: dict[str, tuple] = {
    # 손익
    "revenue": (IS_, ["ifrs-full_Revenue"],
                [r"^매출액$", r"^수익\(매출액\)$", r"^영업수익$", r"^매출$", r"^(총)?매출액", r"^수익$"], None, False),
    "op_income": (IS_, ["dart_OperatingIncomeLoss"],
                  [r"^영업이익", r"^영업손익", r"^영업손실"], None, False),
    "net_income": (IS_, ["ifrs-full_ProfitLoss"],
                   [r"^(연결)?(당기|분기|반기)순(이익|손익|손실)"], r"귀속|지배|비지배", False),
    "ni_owner": (IS_, ["ifrs-full_ProfitLossAttributableToOwnersOfParent"],
                 [r"지배기업.*(소유주|지분)", r"지배주주"], r"비지배|포괄", False),
    "interest_exp": (IS_, ["ifrs-full_InterestExpense", "dart_InterestExpense"],
                     [r"^이자비용"], None, False),
    "fin_cost": (IS_, ["ifrs-full_FinanceCosts"], [r"^금융(비용|원가)"], None, False),
    # 재무상태
    "total_assets": (BS_, ["ifrs-full_Assets"], [r"^자산총계$"], None, False),
    "current_assets": (BS_, ["ifrs-full_CurrentAssets"], [r"^유동자산"], None, False),
    "current_liab": (BS_, ["ifrs-full_CurrentLiabilities"], [r"^유동부채"], None, False),
    "total_liab": (BS_, ["ifrs-full_Liabilities"], [r"^부채총계$"], None, False),
    "equity": (BS_, ["ifrs-full_Equity"], [r"^자본총계$"], None, False),
    "equity_owner": (BS_, ["ifrs-full_EquityAttributableToOwnersOfParent"],
                     [r"지배기업.*(소유주|지분)", r"지배주주"], r"비지배", False),
    "inventory": (BS_, ["ifrs-full_Inventories"], [r"^재고자산$"], None, False),
    "cash": (BS_, ["ifrs-full_CashAndCashEquivalents"], [r"^현금및현금성자산$"], None, False),
    "st_fin": (BS_, ["dart_ShortTermDepositsNotClassifiedAsCashEquivalents"],
               [r"^단기금융(상품|자산)"], None, False),
    "debt": (BS_, [], [r"차입금", r"사채$", r"^유동성장기부채$", r"^(전환|교환|신주인수권부)사채"],
             r"할인|할증|이자|조정|전환권|상환권|파생|충당|미지급", True),
    "lease": (BS_, [], [r"리스부채"], None, True),
    "associates": (BS_, ["ifrs-full_InvestmentsInAssociatesAccountedForUsingEquityMethod",
                         "ifrs-full_InvestmentAccountedForUsingEquityMethod"],
                   [r"관계기업", r"공동기업", r"지분법"], r"손익|처분|손상", False),
    "inv_property": (BS_, ["ifrs-full_InvestmentProperty"], [r"^투자부동산$"], None, False),
    "payables": (BS_, [], [r"^매입채무", r"^(장기)?매입채무및기타채무"], r"미지급", True),
    "advances": (BS_, [], [r"^선수금", r"계약부채", r"^초과청구공사"], None, True),
    "retained": (BS_, ["ifrs-full_RetainedEarnings"], [r"^이익잉여금", r"^결손금"], None, False),
    # 현금흐름
    "da": (CF_, [], [r"감가상각비", r"상각비"], r"환입|손상|대손", True),
    "ocf": (CF_, ["ifrs-full_CashFlowsFromUsedInOperatingActivities"],
            [r"^영업활동.*현금흐름"], None, False),
    "capex": (CF_, ["ifrs-full_PurchaseOfPropertyPlantAndEquipment",
                    "ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"],
              [r"^유형자산의?\s*취득"], None, False),
    "interest_paid": (CF_, ["ifrs-full_InterestPaidClassifiedAsOperatingActivities",
                            "ifrs-full_InterestPaidClassifiedAsFinancingActivities"],
                      [r"^이자의?지급"], None, False),
    "dividends_paid": (CF_, ["ifrs-full_DividendsPaidClassifiedAsFinancingActivities"],
                       [r"^배당금의?지급"], r"비지배", False),
    "buyback": (CF_, ["dart_AcquisitionOfTreasuryShares"], [r"^자기주식의?취득"], None, False),
    "share_issue": (CF_, ["ifrs-full_ProceedsFromIssuingShares"],
                    [r"^유상증자", r"^주식의?발행"], r"비용|비지배", False),
}

LABELS = {
    "revenue": "매출액", "op_income": "영업이익", "net_income": "당기순이익",
    "ni_owner": "지배주주순이익", "interest_exp": "이자비용", "fin_cost": "금융비용",
    "total_assets": "자산총계", "current_assets": "유동자산", "current_liab": "유동부채",
    "total_liab": "부채총계", "equity": "자본총계", "equity_owner": "지배주주지분",
    "inventory": "재고자산", "cash": "현금및현금성자산", "st_fin": "단기금융상품",
    "debt": "총차입금", "lease": "리스부채", "associates": "관계·공동기업투자",
    "inv_property": "투자부동산", "da": "감가상각비(D&A)", "ocf": "영업활동현금흐름",
    "capex": "CAPEX(유형자산취득)", "interest_paid": "이자의 지급",
    "payables": "매입채무", "advances": "선수금·계약부채", "retained": "이익잉여금",
    "dividends_paid": "배당금 지급", "buyback": "자기주식 취득", "share_issue": "유상증자",
}
FLOW_KEYS = [k for k, v in ITEMS.items() if v[0] != BS_]
STOCK_KEYS = [k for k, v in ITEMS.items() if v[0] == BS_]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s))


def _contains(names: pd.Series, pattern: str) -> pd.Series:
    rx = re.compile(pattern)
    return names.map(lambda s: rx.search(s) is not None).astype(bool)


def _match_rows(df: pd.DataFrame, key: str) -> pd.DataFrame:
    sj, ids, patterns, exclude, summing = ITEMS[key]
    sub = df[df["sj_div"].isin(sj)]
    if sub.empty:
        return sub
    # 손익 항목은 IS 를 CIS 보다 우선
    sub = sub.assign(_ord=sub["sj_div"].map({s: i for i, s in enumerate(sj)}))
    sub = sub.sort_values("_ord", kind="stable")
    names = sub["account_nm"].map(_norm)

    if ids and not summing:
        hit = sub[sub["account_id"].isin(ids)]
        if exclude:
            hit = hit[~_contains(names[hit.index], exclude)]
        if not hit.empty:
            return hit.iloc[[0]]

    excluded = _contains(names, exclude) if exclude else pd.Series(False, index=sub.index)
    mask = pd.Series(False, index=sub.index)
    for pat in patterns:
        m = _contains(names, pat) & ~excluded
        if not summing and m.any():
            mask = m
            break
        mask |= m
    hit = sub[mask]
    if hit.empty:
        return hit
    if not summing:
        return hit.iloc[[0]]
    # 같은 계정명·같은 금액이 두 번 잡히는 경우(소계 중복)만 제거.
    # 유동/비유동 '차입금'처럼 이름만 같고 금액이 다르면 둘 다 더한다.
    dup = pd.DataFrame({"nm": names[hit.index], "amt": hit.get("thstrm_amount")}).duplicated()
    return hit.loc[~dup]


def _value_cols(sj: str, period: str, quarterly: bool) -> list[str]:
    if quarterly and sj in IS_:
        # 분기/반기 손익은 누적(add) 금액을 사용
        return {
            "cur": ["thstrm_add_amount", "thstrm_amount"],
            "prev": ["frmtrm_add_amount", "frmtrm_q_amount", "frmtrm_amount"],
            "prev2": [],
        }[period]
    return {"cur": ["thstrm_amount"], "prev": ["frmtrm_amount"], "prev2": ["bfefrmtrm_amount"]}[period]


def get(df: pd.DataFrame, key: str, period: str = "cur", quarterly: bool = False) -> float | None:
    if df is None or df.empty:
        return None
    rows = _match_rows(df, key)
    if rows.empty:
        return None
    total, found = 0.0, False
    for _, row in rows.iterrows():
        for col in _value_cols(row["sj_div"], period, quarterly):
            v = to_num(row.get(col))
            if v is not None:
                total += v
                found = True
                break
    return total if found else None


@dataclass
class Analysis:
    corp_name: str
    report: Report
    fs_div: str
    history: pd.DataFrame              # index=연도, 매출/영업이익/순이익 (연간)
    flows: dict                         # 밸류에이션용 손익·현금흐름 (연간 또는 TTM)
    flows_basis: str                    # "FY2024" / "TTM(2025.06)"
    stocks: dict                        # 보고서 기말 재무상태
    ytd: pd.DataFrame | None = None     # 분기 보고서일 때 누적 실적 비교
    raw: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)
    stocks_prev: dict = field(default_factory=dict)   # 직전 연말 재무상태
    since_flows: dict = field(default_factory=dict)   # 직전 연말 이후 누적 손익·현금흐름


def _annual_history(client: DartClient, corp_code: str, year: int, df_y: pd.DataFrame,
                    fs_div: str) -> pd.DataFrame:
    keys = ["revenue", "op_income", "net_income", "ni_owner"]
    data = {year: {}, year - 1: {}, year - 2: {}}
    for k in keys:
        data[year][k] = get(df_y, k, "cur")
        data[year - 1][k] = get(df_y, k, "prev")
        data[year - 2][k] = get(df_y, k, "prev2")
    # 3기 전 금액이 없으면 전년도 사업보고서의 전기 금액으로 채운다
    if any(data[year - 2][k] is None for k in keys[:3]):
        df_p = client.financial_statements(corp_code, str(year - 1), REPRT_ANNUAL, fs_div)
        for k in keys:
            if data[year - 2][k] is None:
                data[year - 2][k] = get(df_p, k, "prev")
    hist = pd.DataFrame(data).T.sort_index()
    hist.index = [f"FY{y}" for y in hist.index]
    return hist.dropna(how="all")


def analyze(client: DartClient, corp_code: str, corp_name: str, rep: Report,
            fs_div: str = "CFS") -> Analysis:
    warnings: list[str] = []
    cur = client.financial_statements(corp_code, rep.bsns_year, rep.reprt_code, fs_div)
    if cur.empty:
        raise ValueError("해당 보고서의 재무제표 데이터가 DART API에 없습니다. (금융업·신규상장 등은 미제공일 수 있음)")
    used_div = cur.attrs.get("fs_div", fs_div)
    if used_div != fs_div:
        warnings.append("연결재무제표가 없어 별도재무제표로 분석했습니다.")

    stocks = {k: get(cur, k, "cur") for k in STOCK_KEYS}

    if rep.is_annual:
        year = int(rep.bsns_year)
        annual = cur
        flows = {k: get(cur, k, "cur") for k in FLOW_KEYS}
        basis = f"FY{year} 연간"
        ytd = None
    else:
        year = int(rep.bsns_year) - 1
        annual = client.financial_statements(corp_code, str(year), REPRT_ANNUAL, used_div)
        flows, ytd_rows = {}, {}
        prior = None  # 전년 동기 보고서 (분기 현금흐름표엔 전기 금액이 없는 경우가 많음)
        for k in FLOW_KEYS:
            a = get(annual, k, "cur")
            c = get(cur, k, "cur", quarterly=True)
            p = get(cur, k, "prev", quarterly=True)
            if p is None and c is not None:
                if prior is None:
                    prior = client.financial_statements(corp_code, str(year), rep.reprt_code, used_div)
                p = get(prior, k, "cur", quarterly=True)
            ytd_rows[k] = (c, p)
            # TTM = 직전 연간 + 당기 누적 - 전기 누적
            flows[k] = None if a is None or c is None or p is None else a + c - p
            if flows[k] is None and k in ("revenue", "op_income", "net_income", "ni_owner"):
                warnings.append(f"{LABELS[k]} TTM 계산 불가 → 직전 연간 값 사용")
                flows[k] = a
        basis = f"TTM ({rep.period_end} 기준 최근 4개 분기)"
        ytd = pd.DataFrame(
            {"당기 누적": {LABELS[k]: v[0] for k, v in ytd_rows.items()},
             "전년 동기 누적": {LABELS[k]: v[1] for k, v in ytd_rows.items()}}
        ).loc[[LABELS[k] for k in ("revenue", "op_income", "net_income", "ni_owner")]].astype(float)
        ytd["YoY(%)"] = (ytd["당기 누적"] - ytd["전년 동기 누적"]) / ytd["전년 동기 누적"].abs() * 100

    if flows.get("ni_owner") is None:
        flows["ni_owner"] = flows.get("net_income")
        warnings.append("지배주주순이익 계정이 없어 당기순이익을 사용했습니다.")
    if stocks.get("equity_owner") is None:
        stocks["equity_owner"] = stocks.get("equity")
    if flows.get("da") is None:
        warnings.append("현금흐름표에서 감가상각비를 찾지 못했습니다. EBITDA = 영업이익으로 계산됩니다 ('고급 설정'에서 직접 입력 가능).")
    if flows.get("interest_exp") is None:
        if flows.get("interest_paid") is not None:
            flows["interest_exp"] = flows["interest_paid"]
            warnings.append("손익계산서에 이자비용 계정이 없어 현금흐름표의 '이자의 지급'으로 이자보상배율을 계산했습니다.")
        elif flows.get("fin_cost") is not None:
            flows["interest_exp"] = flows["fin_cost"]
            warnings.append("이자비용 계정이 없어 금융비용 전체(외환손실 포함)로 이자보상배율을 계산했습니다.")

    history = _annual_history(client, corp_code, year, annual, used_div) if not annual.empty \
        else pd.DataFrame()

    # 부채비율 변화 분해용: 재무상태표 전기(=직전 연말)와, 그 이후 누적(분기면 YTD) 손익·현금흐름
    stocks_prev = {k: get(cur, k, "prev") for k in STOCK_KEYS}
    since_flows = {k: get(cur, k, "cur", quarterly=not rep.is_annual) for k in FLOW_KEYS}

    return Analysis(
        corp_name=corp_name, report=rep, fs_div=used_div, history=history,
        flows=flows, flows_basis=basis, stocks=stocks, ytd=ytd, raw=cur, warnings=warnings,
        stocks_prev=stocks_prev, since_flows=since_flows,
    )


QUARTER_KEYS = ["revenue", "op_income", "net_income", "ni_owner"]
_QUARTER_ORDER = [(REPRT_Q1, 1), (REPRT_HALF, 2), (REPRT_Q3, 3), (REPRT_ANNUAL, 4)]


def quarterly_history(client: DartClient, corp_code: str, latest: Report, fs_div: str,
                      years: int = 3) -> pd.DataFrame:
    """분기 단독 실적. 1~3분기는 보고서의 3개월 금액, 4분기 = 연간 − 3분기 누적."""
    last_year = int(latest.bsns_year)
    last_q = dict(_QUARTER_ORDER)[latest.reprt_code]
    rows = {}
    for y in range(last_year - years + 1, last_year + 1):
        q3_cum = None
        for code, q in _QUARTER_ORDER:
            if y == last_year and q > last_q:
                break
            df = client.financial_statements(corp_code, str(y), code, fs_div)
            if df.empty:
                continue
            if q == 4:
                annual = {k: get(df, k, "cur") for k in QUARTER_KEYS}
                vals = {k: None if annual[k] is None or q3_cum is None or q3_cum.get(k) is None
                        else annual[k] - q3_cum[k] for k in QUARTER_KEYS}
            else:
                vals = {k: get(df, k, "cur") for k in QUARTER_KEYS}  # thstrm_amount = 3개월
                if q == 3:
                    q3_cum = {k: get(df, k, "cur", quarterly=True) for k in QUARTER_KEYS}
            rows[f"{str(y)[2:]}.{q}Q"] = vals
    return pd.DataFrame(rows).T.astype(float)


def annual_series(client: DartClient, corp_code: str, last_year: int, fs_div: str,
                  years: int = 6) -> pd.DataFrame:
    """밴드 차트용 연간 지배주주순이익·지배주주지분·매출 (사업보고서 2년치씩 조회)."""
    keys = ["ni_owner", "net_income", "equity_owner", "equity", "revenue", "total_liab"]
    data: dict[int, dict] = {}
    for y in range(last_year, last_year - years, -2):
        df = client.financial_statements(corp_code, str(y), REPRT_ANNUAL, fs_div)
        if df.empty:
            continue
        for fy, period in ((y, "cur"), (y - 1, "prev")):
            data.setdefault(fy, {k: get(df, k, period) for k in keys})
    out = pd.DataFrame(data).T.sort_index().astype(float)
    if out.empty:
        return out
    out["ni_owner"] = out["ni_owner"].fillna(out["net_income"])
    out["equity_owner"] = out["equity_owner"].fillna(out["equity"])
    return out[["ni_owner", "equity_owner", "revenue", "total_liab", "equity"]]
