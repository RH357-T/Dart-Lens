"""OpenDART API 클라이언트.

사용하는 API
- corpCode.xml        : 회사 고유번호(corp_code) 전체 목록 (zip)
- list.json           : 공시 검색 (정기공시: 사업/반기/분기보고서)
- fnlttSinglAcntAll.json : 단일회사 전체 재무제표
- stockTotqySttus.json   : 주식의 총수 현황
"""
from __future__ import annotations

import io
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import requests

BASE = "https://opendart.fss.or.kr/api"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")

REPRT_ANNUAL = "11011"  # 사업보고서
REPRT_HALF = "11012"    # 반기보고서
REPRT_Q1 = "11013"      # 1분기보고서
REPRT_Q3 = "11014"      # 3분기보고서

REPRT_LABEL = {
    REPRT_ANNUAL: "사업보고서",
    REPRT_HALF: "반기보고서",
    REPRT_Q1: "1분기보고서",
    REPRT_Q3: "3분기보고서",
}

# DART 응답 status 코드
STATUS_OK = "000"
STATUS_NO_DATA = "013"


class DartError(RuntimeError):
    pass


@dataclass
class Report:
    rcept_no: str
    report_nm: str
    rcept_dt: str
    bsns_year: str
    reprt_code: str
    period_end: str  # YYYY.MM

    @property
    def label(self) -> str:
        return f"{self.report_nm.strip()}  ·  접수 {self.rcept_dt}"

    @property
    def is_annual(self) -> bool:
        return self.reprt_code == REPRT_ANNUAL

    @property
    def url(self) -> str:
        return f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={self.rcept_no}"


class DartClient:
    def __init__(self, api_key: str, timeout: int = 20):
        if not api_key:
            raise DartError("DART API 키가 필요합니다.")
        self.api_key = api_key.strip()
        self.timeout = timeout
        self.session = requests.Session()

    # ------------------------------------------------------------------ 공통
    def _request(self, endpoint: str, params: dict, timeout: int) -> requests.Response:
        # requests 예외 메시지엔 crtfc_key 가 든 URL 이 포함되므로, 화면에 키가 노출되지 않게 바꿔 던진다
        try:
            r = self.session.get(f"{BASE}/{endpoint}", params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            status = getattr(e.response, "status_code", None)
            raise DartError(f"DART 서버 접속 오류 ({type(e).__name__}{f', HTTP {status}' if status else ''})") \
                from None

    def _get_json(self, endpoint: str, **params) -> dict:
        params["crtfc_key"] = self.api_key
        try:
            data = self._request(endpoint, params, self.timeout).json()
        except ValueError:  # 점검 중 안내 페이지 등 JSON 이 아닌 응답
            raise DartError("DART 응답 형식 오류 (서버 점검 중일 수 있습니다)") from None
        status = data.get("status")
        if status not in (STATUS_OK, STATUS_NO_DATA):
            raise DartError(f"[{status}] {data.get('message')}")
        return data

    # ------------------------------------------------------------ 회사 코드
    def corp_codes(self, refresh: bool = False) -> pd.DataFrame:
        """상장사 고유번호 목록. 최초 1회 내려받아 .cache 에 저장한다."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, "corp_codes.csv")
        if os.path.exists(path) and not refresh:
            return pd.read_csv(path, dtype=str).fillna("")

        r = self._request("corpCode.xml", {"crtfc_key": self.api_key}, 60)
        try:
            zf = zipfile.ZipFile(io.BytesIO(r.content))
        except zipfile.BadZipFile:
            # 키 오류 등은 zip 대신 JSON/XML 에러로 응답한다
            raise DartError(r.content[:300].decode("utf-8", "ignore"))
        root = ET.fromstring(zf.read(zf.namelist()[0]))
        rows = [
            {
                "corp_code": el.findtext("corp_code", "").strip(),
                "corp_name": el.findtext("corp_name", "").strip(),
                "stock_code": el.findtext("stock_code", "").strip(),
            }
            for el in root.iter("list")
        ]
        df = pd.DataFrame(rows)
        df = df[df["stock_code"] != ""].reset_index(drop=True)
        df.to_csv(path, index=False, encoding="utf-8")
        return df

    def search_corp(self, keyword: str) -> pd.DataFrame:
        df = self.corp_codes()
        kw = keyword.strip()
        if not kw:
            return df.iloc[0:0]
        if kw.isdigit():
            hit = df[df["stock_code"].str.contains(kw)]
        else:
            hit = df[df["corp_name"].str.contains(kw, case=False, regex=False)]
        # 정확히 일치하는 이름을 맨 위로
        hit = hit.assign(_exact=(hit["corp_name"] == kw)).sort_values(
            ["_exact", "corp_name"], ascending=[False, True]
        )
        return hit.drop(columns="_exact").head(30)

    # ------------------------------------------------------------ 공시 검색
    def periodic_reports(self, corp_code: str, years: int = 4) -> list[Report]:
        """정기공시(사업/반기/분기보고서) 목록. 정정공시는 최신본만 남긴다."""
        end = date.today()
        bgn = end - timedelta(days=365 * years)
        data = self._get_json(
            "list.json",
            corp_code=corp_code,
            bgn_de=bgn.strftime("%Y%m%d"),
            end_de=end.strftime("%Y%m%d"),
            pblntf_ty="A",
            page_count=100,
        )
        items = data.get("list", [])

        # 결산월 추정: 사업보고서의 (YYYY.MM) 에서 MM
        fy_end_month = 12
        for it in items:
            m = re.search(r"사업보고서\s*\((\d{4})\.(\d{2})\)", it["report_nm"])
            if m:
                fy_end_month = int(m.group(2))
                break

        seen: dict[tuple, Report] = {}
        for it in items:
            nm = it["report_nm"]
            m = re.search(r"(사업|반기|분기)보고서\s*\((\d{4})\.(\d{2})\)", nm)
            if not m:
                continue
            kind, yyyy, mm = m.group(1), int(m.group(2)), int(m.group(3))
            if kind == "사업":
                code = REPRT_ANNUAL
            elif kind == "반기":
                code = REPRT_HALF
            else:
                offset = (mm - fy_end_month) % 12
                code = REPRT_Q1 if offset <= 3 else REPRT_Q3
            # 사업연도: 결산월이 12월이 아니면 기말이 속한 해 기준으로 보정
            bsns_year = yyyy if (fy_end_month == 12 or mm <= fy_end_month) else yyyy + 1
            if code == REPRT_ANNUAL:
                bsns_year = yyyy
            key = (bsns_year, code)
            rep = Report(
                rcept_no=it["rcept_no"],
                report_nm=nm,
                rcept_dt=it["rcept_dt"],
                bsns_year=str(bsns_year),
                reprt_code=code,
                period_end=f"{yyyy}.{mm:02d}",
            )
            # list.json 은 최신 접수순 → 처음 본 것이 최신(정정 반영)본
            seen.setdefault(key, rep)
        return sorted(seen.values(), key=lambda r: (r.period_end, r.rcept_dt), reverse=True)

    # ------------------------------------------------------------ 재무제표
    def financial_statements(
        self, corp_code: str, bsns_year: str, reprt_code: str, fs_div: str = "CFS"
    ) -> pd.DataFrame:
        """전체 재무제표. 연결(CFS)이 없으면 별도(OFS)로 재시도."""
        for div in ([fs_div, "OFS"] if fs_div == "CFS" else [fs_div]):
            data = self._get_json(
                "fnlttSinglAcntAll.json",
                corp_code=corp_code,
                bsns_year=bsns_year,
                reprt_code=reprt_code,
                fs_div=div,
            )
            rows = data.get("list") or []
            if rows:
                df = pd.DataFrame(rows)
                df.attrs["fs_div"] = div
                return df
        return pd.DataFrame()

    def company(self, corp_code: str) -> dict:
        """기업개황 (induty_code: 표준산업분류 코드 등)."""
        return self._get_json("company.json", corp_code=corp_code)

    def share_count(self, corp_code: str, bsns_year: str, reprt_code: str) -> dict:
        """보통주 발행주식총수 / 자기주식 / 유통주식수."""
        data = self._get_json(
            "stockTotqySttus.json",
            corp_code=corp_code,
            bsns_year=bsns_year,
            reprt_code=reprt_code,
        )
        out = {"issued": None, "treasury": None, "floating": None}
        for row in data.get("list", []):
            # '보통주' 또는 '의결권 있는 주식\n(보통주)' 처럼 회사마다 표기가 다르다
            if "보통주" in re.sub(r"\s+", "", row.get("se", "")):
                out["issued"] = to_num(row.get("istc_totqy"))
                out["treasury"] = to_num(row.get("tesstk_co"))
                out["floating"] = to_num(row.get("distb_stock_co"))
                break
        return out


def to_num(v) -> float | None:
    if v is None:
        return None
    s = str(v).replace(",", "").strip()
    if s in ("", "-"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        x = float(s)
    except ValueError:
        return None
    if x != x:  # NaN (빈 컬럼)
        return None
    return -x if neg else x
