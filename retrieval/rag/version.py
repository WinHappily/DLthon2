"""A-6 : 기준일 · 판본 규칙 (v2).

항목 파악(표 07) → 기준일 종류 → 질문 속 날짜마다 '무슨 날짜인지(역할)' → 기준일에 맞는 날짜 고르기
→ 부칙 제1조 시행 구간과 비교. 맞는 날짜가 없으면 '추가 확인'(제8조).

    python -m rag.version "생신 식사와 케이크 24만 원, 첫 결제 9월 30일"
    python -m rag.version "..." --doc V3

v1 → v2 에서 바꾼 것 (eval/questions_version.jsonl 30문항 : v1 19/30)
  1. 날짜 역할 : 날짜 뒤(없으면 바로 앞) 구절의 낱말로 결제 · 주문 · 예약 · 시작 · 종료 · 고지 · 환불 ... 을 붙이고,
     기준일 종류(표 07 문구)에서 받아들일 역할을 정한다. 고지일 · 문의일 · 첫 매수일 · 배송일은 기준일이 아니다
  2. 시각 : '9월 30일 24시' · '자정'은 10월 1일 00시 (표 53, 경계 시각은 새 판본)
  3. 상대 날짜 : '전날' · '다음 날' · '같은 날'은 앞에 나온 날짜 기준
  4. 기준일이 아닌 날짜만 있으면 (예 : 행사일만 있고 첫 결제일이 없음) '추가 확인'
  5. 개정 항목이 아니어도 적용 판본을 채운다 : 날짜가 있으면 그 날짜의 판본, 없으면 그 문서의 현행 판본
  6. 항목이 여럿이면 항목마다 따로 판정(by_item), 주 항목은 맞은 키워드가 많은 것
경과조치 : 후속 결제는 첫 확정 결제일(제10조), 환불은 원거래(제10조 · 제26조), 전량 취소 · 확정 취소 후
새 주문 · 예약 · 숙박은 새 날짜(부칙 제4조 · 제7조 · 제9조), 색상 · 배송일만 바꾸면 원 주문일(부칙 제4조).

문서마다 '아는 판본'이 다르다. V2 문서로 답하면 V3 시행 뒤 날짜도 V2로 판정한다(그 문서엔 V3가 없으므로).
문서별 설정(config/versions.json 의 docs.<doc>) : versions · rule_pages · compare_pages.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date, timedelta

from .common import load_config

# 날짜 + (선택) 시각.  '9월 30일 23시 50분', '10월 1일 0시 5분', '9월 30일 저녁 7시', '9월 30일 24시', '9월 30일 자정'
_TIME = r"(?:\s*(?:오전|오후|저녁|밤|아침|새벽|낮)?\s*(?:(\d{1,2})\s*시(?:\s*\d{1,2}\s*분|\s*반)?|(자정)))?"
DATE_PATTERNS = [
    re.compile(r"(?:(\d{4})\s*년\s*)?(\d{1,2})\s*월\s*(\d{1,2})\s*일" + _TIME),
    re.compile(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})" + _TIME),
    re.compile(r"(?<![\d.])()(\d{1,2})/(\d{1,2})(?![\d/])" + _TIME),
]
# 일자 없는 달 : '2026년 11월', '11월에' (뒤에 일이 오면 위 패턴이 잡는다). '이번 달'은 문서 설정 this_month
MONTH_ONLY = re.compile(r"(?:(\d{4})\s*년\s*)?(\d{1,2})\s*월(?!\s*\d{1,2}\s*일)(?!\s*\d)")
THIS_MONTH = re.compile(r"이번\s*달")
# 월 없는 일자 '17일' (앞 날짜의 달). '3일간' 같은 기간은 뺀다
DAY_ONLY = re.compile(r"(?<![\d월/.\-])(\d{1,2})\s*일(?!\s*(?:간|전|후|뒤|동안|째|이내|안에|만에|마다))(?=[\s,은는이에도까지부터(])")
# 질문이 판본을 직접 지정 : 'V2에서', 'V2 평일 …'. 'V2 합본'은 문서 이름이라 제외
NAMED_VERSION = re.compile(r"(?<![A-Za-z])(V\d+)(?!\s*합본)(?![\d])")
RELATIVE = re.compile(r"(전날|전일|하루\s*전|다음\s*날|이튿날|익일|같은\s*날)" + _TIME)
REL_DAYS = {"전날": -1, "전일": -1, "하루전": -1, "다음날": 1, "이튿날": 1, "익일": 1, "같은날": 0}

# 날짜 뒤 구절을 끊는 곳 (한국어는 동사가 구절 끝에 온다)
# '10.5%' · '1,200만 원'처럼 숫자 사이의 점 · 쉼표는 구절 끝이 아니다 (D097)
CLAUSE_END = re.compile(r"(?<!\d)[.,]|[.,](?!\d)|[!?\n]|고\s|고$|는데|지만|다가|면서|며\s|서\s|라\s|니까|인데")

ROLE_WORDS = [  # (역할, 정규식). 한 구절에 여러 역할이 붙을 수 있다
    ("추가주문", r"추가\s*매수|추가\s*주문|물타기"),
    ("결제", r"결제|계산|지불|카드로|구입|구매|샀|사서|냈|썼|잔금|예약금"),
    ("주문", r"주문"),
    ("확정", r"확정"),
    ("예약", r"예약|예매|부킹"),
    ("시작", r"^\s*(?:에\s*)?부터|시작|출발|입실|체크인|출근"
             r"|\d{1,2}\s*시(?:\s*\d+\s*분)?\s*(?:부터|~)\s*\d{1,2}\s*시"),   # '토 친구 4명 19시~22시 예정' : 그날 시작하는 일 (S0513 · S0774)
    ("숙박", r"숙박|묵|자고|재워|\d+\s*박"),
    ("방문", r"방문|놀러|(?<![나들])와서|오기로|온다|오셔|오시|(?:있|잇|놀)다\s*(?:가|갈|간|갔)"),   # '22시까지 있다 갈 건데' (S0757)
    ("초대", r"초대"),
    ("가사", r"배출|당번|청소|설거지|빨래"),
    ("종료", r"끝|종료|퇴근|체크아웃|퇴실"),
    ("고지", r"알렸|알려|알린|고지|공지|말했|메시지|연락"),               # '감석한테 알린 날' (S0584)
    ("환불", r"환불|환급"),
    ("취소", r"취소"),
    ("문의", r"문의|알아보|알아봤|알아봐|장바구니|견적|상담|비교"),
    ("행사", r"생신|생일|명절|결혼식|장례|기일|행사"),
    ("배송", r"배송|도착|받았|받는"),
    ("변경", r"바꿨|바꾸|변경"),
    ("체결", r"체결(?!가)"),
    ("심사", r"심사|대출\s*신청"),                                        # V3 주택 월 상환 한도 : 매수자금 심사일
    ("모임", r"술자리|음주|회식|술\s*약속|한잔|동창회|모임"),               # V3 금·토 술자리 귀가 : 음주 모임 시작일
    ("근무", r"야간\s*근무|밤\s*근무|밤샘|당직"),                          # '11-20 금 밤 근무 06시 종료' : 근무한 날 = 근무 시작일 (S0434)
]
ROLE_RES = [(r, re.compile(p)) for r, p in ROLE_WORDS]


def accepted_roles(basis: str) -> set[str]:
    """표 07의 기준일 문구 → 기준일로 받아들일 날짜 역할. 새 판본의 새 항목에도 문구만으로 동작한다."""
    acc = set()
    if "결제" in basis:
        acc |= {"결제", "확정"}
    if "주문" in basis:
        acc |= {"추가주문"} if "추가" in basis else {"주문", "확정", "결제"}   # 온라인 구매는 결제 = 주문 확정
    if "예약" in basis:
        acc |= {"예약", "확정"}
    if "숙박" in basis:
        acc |= {"시작", "숙박"}
    if "방문" in basis:
        acc |= {"시작", "방문"}
    if "가사" in basis:
        acc |= {"시작", "가사"}
    if "시작" in basis:
        acc.add("시작")
    if "심사" in basis:
        acc.add("심사")
    if "모임" in basis or "음주" in basis:
        acc.add("모임")
    if "근무" in basis:
        acc.add("근무")
    return acc or {"결제", "확정", "주문", "예약", "시작"}


SAME_DAY_NEW = re.compile(r"취소[^.]{0,30}(새|다시|재|별도)[^.]{0,20}(주문|예약|확정|잡)")
# 항목 찾기 전에 지우는 부정 표현 : '자고 가는 건 아니야', '숙박 규칙은 상관없고' (D081)
NEG_STAY = re.compile(r"자고\s*가는\s*(?:건|게|것은?)\s*아니\S*|자지\s*않\S*|숙박(?:은|이)?\s*아니\S*|숙박\S*\s*(?:규칙은\s*)?상관없\S*"
                      r"|숙박\s*(?:은|이)?\s*없\S*")   # '(숙박 없음)' (S0728)
SEPARATE = re.compile(r"따로|별도|추가로\s*예약|연장")
GENERAL_ROLES = {"결제", "확정", "주문", "예약", "시작", "방문", "숙박", "가사"}  # 제9조 일반 기준일
NEW_TXN = re.compile(r"(전량|전부|완전히|확정)\s*취소|취소[^.]{0,40}(새|다시|재|별도|다른)"
                     r"|새\s*(주문|예약)|새로\s*(주문|예약|잡|확정|묵)|다시\s*(주문|예약|잡)|재주문|재예약"
                     r"|별도\s*(주문|예약)|새\s*주문\s*번호|다른\s*(숙소|호텔|날짜)[^.]{0,15}(예약|잡|확정)")


def _date_at(m, default_year: int) -> date | None:
    y = int(m.group(1)) if m.group(1) else default_year
    try:
        d = date(y, int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None
    if (m.group(4) and int(m.group(4)) == 24) or m.group(5):
        d += timedelta(days=1)   # 24시 · 자정 = 다음 날 00시
    return d


def find_dates(text: str, default_year: int, this_month: str | None = None) -> list[dict]:
    """[{date, start, end, rel}] 글자 위치 순. 상대 날짜는 가장 가까운 앞 날짜(없으면 뒤 날짜) 기준."""
    out, used = [], []
    for pat in DATE_PATTERNS:
        for m in pat.finditer(text):
            if any(a <= m.start() < b for a, b in used):
                continue
            d = _date_at(m, default_year)
            if d:
                out.append({"date": d, "start": m.start(), "end": m.end(), "rel": None})
                used.append((m.start(), m.end()))
    for m in DAY_ONLY.finditer(text):     # '16일이랑 17일' : 월 없는 일자는 바로 앞 날짜의 달 (D067)
        if any(a <= m.start() < b for a, b in used):
            continue
        prev = [x for x in out if x["end"] <= m.start() and not x.get("month")]
        if not prev:
            continue
        ref = max(prev, key=lambda x: x["end"])["date"]
        try:
            d = ref.replace(day=int(m.group(1)))
        except ValueError:
            continue
        out.append({"date": d, "start": m.start(), "end": m.end(), "rel": None})
        used.append((m.start(), m.end()))
    for m in MONTH_ONLY.finditer(text):   # 일자 없는 달 : 그달 1일로 두고 month 표시
        if any(a <= m.start() < b for a, b in used):
            continue
        try:
            d = date(int(m.group(1)) if m.group(1) else default_year, int(m.group(2)), 1)
        except ValueError:
            continue
        out.append({"date": d, "start": m.start(), "end": m.end(), "rel": None, "month": True})
        used.append((m.start(), m.end()))
    out.sort(key=lambda x: x["start"])
    rels = []
    for m in RELATIVE.finditer(text):
        if any(a <= m.start() < b for a, b in used) or not out:
            continue
        before = [x for x in out if x["start"] < m.start()]
        ref = before[-1] if before else out[0]
        d = ref["date"] + timedelta(days=REL_DAYS[re.sub(r"\s+", "", m.group(1))])
        if (m.group(2) and int(m.group(2)) == 24) or m.group(3):
            d += timedelta(days=1)
        rels.append({"date": d, "start": m.start(), "end": m.end(), "rel": m.group(1)})
    if this_month:
        for m in THIS_MONTH.finditer(text):
            if not any(a <= m.start() < b for a, b in used):
                rels.append({"date": date.fromisoformat(this_month + "-01"), "start": m.start(), "end": m.end(),
                             "rel": None, "month": True})
    return sorted(out + rels, key=lambda x: x["start"])


def tag_roles(text: str, dates: list[dict]) -> None:
    """날짜마다 roles 를 붙인다. 날짜 바로 뒤 구절 → 없으면 바로 앞 구절."""
    for i, x in enumerate(dates):
        nxt = dates[i + 1]["start"] if i + 1 < len(dates) else len(text)
        after = text[x["end"]:nxt]
        if re.match(r"\s*(?:에\s*)?까지", after):        # '10월 2일까지' 는 끝나는 날
            x.update(roles={"종료"}, clause=after[:20].strip())
            continue
        m = CLAUSE_END.search(after)
        clause = after[:m.end()] if m else after
        roles = {r for r, rx in ROLE_RES if rx.search(clause)}
        if not roles:
            prev = dates[i - 1]["end"] if i else 0
            before = text[max(prev, x["start"] - 14):x["start"]]
            before = CLAUSE_END.split(before)[-1]
            roles = {r for r, rx in ROLE_RES if r != "시작" and rx.search(before)}
            if re.search(r"시작|숙박은|방문은", before):
                roles.add("시작")
            clause = before + "·" + clause
        if "근무" in roles:   # '07시에 끝나는 야간근무' · '06시 30분까지 야간근무'의 날짜는 끝난 날 (D064 · S0426), '밤 근무 06시 종료'는 근무한 날 (S0434)
            w = re.search(dict(ROLE_WORDS)["근무"], clause)
            e = re.search(dict(ROLE_WORDS)["종료"] + r"|까지", clause)
            if w and e and e.start() < w.start():
                roles.discard("근무")
        x.update(roles=roles, clause=clause.strip())
    # '10-20에 식사 13만 원, 10-21에 선물 8만 원을 결제했어' : 앞 날짜에 역할이 없고 같은 문장 안 다음 날짜에
    # 역할이 있으면 그 동사를 함께 쓴다 (D055)
    for i in range(len(dates) - 2, -1, -1):
        x, y = dates[i], dates[i + 1]
        between = text[x["end"]:y["start"]]
        if (not x["roles"] and y["roles"] and not x.get("month") and not y.get("month")
                and "," in between and not re.search(r"[.?!\n]|서\s|고\s|며\s|는데|다가|지만|면서", between)):
            x["roles"] = set(y["roles"]) - {"종료"}


def _kw_pat(w: str) -> str:
    """키워드 → 정규식. 're:' 는 그대로, 띄어 쓴 키워드는 띄어쓰기를 무시('내부 대출 상한' = '내부대출상한', V3 N016)."""
    return w[3:] if w.startswith("re:") else r"\s*".join(re.escape(p) for p in w.split())


def _kw(w: str, text: str) -> bool:
    """키워드 하나. 're:' 로 시작하면 정규식 ('하자고' 속 '자고'처럼 다른 말에 섞여 걸리는 짧은 낱말용)."""
    return bool(re.search(_kw_pat(w), text))


def _count(words: list[str], text: str) -> int:
    """맞은 키워드 수. 겹치는 위치에서 맞은 것은 하나로 센다."""
    spans = []
    for w in words:
        pat = _kw_pat(w)
        spans += [(m.start(), m.end()) for m in re.finditer(pat, text) if m.end() > m.start()]
    spans.sort()
    n, end = 0, -1
    for a, b in spans:
        if a >= end:
            n += 1
            end = b
        else:
            end = max(end, b)
    return n


def detect_items(text: str, items: list[dict]) -> list[tuple[dict, int]]:
    """[(항목, 맞은 키워드 수)] 많은 순, 같으면 설정 순."""
    text = NEG_STAY.sub(" ", text)
    dated = any(p.search(text) for p in DATE_PATTERNS)
    hits = []
    for n, it in enumerate(items):
        if any(_kw(w, text) for w in it.get("none", [])):   # 바뀌지 않은 하위 규정을 묻는 말
            continue
        # none_soft : 다과 · 소모품 · 택시비처럼 판본 무관한 비용. 단 판정을 가르는 조건(decisive, 예 : 고지 · 합의가
        # 아직 · 없이)이 열려 있으면 비용을 함께 물어도 그 항목이다 (S0515 · S0657 · S0658 · S0680, C 11.20)
        if (any(_kw(w, text) for w in it.get("none_soft", []))
                and not any(_kw(w, text) for w in it.get("decisive", []))):
            continue
        # any : 하나만 맞아도 그 항목 / all : 묶음마다 하나 이상. 둘 다 있으면 어느 쪽이든 맞으면 된다
        k_any = _count(it.get("any", []), text)
        if dated:   # any_dated : 날짜가 있을 때만 그 항목으로 보는 말 ('원래 예약은 9월 25일 확정 … 취소', S0584)
            k_any += _count(it.get("any_dated", []), text)
        ks = [_count(g, text) for g in it.get("all", [])]
        k_all = sum(ks) if ks and all(ks) else 0
        if k_any or k_all:
            hits.append((it, max(k_any, k_all), n))
    hits.sort(key=lambda x: (-x[1], x[2]))
    return [(it, k) for it, k, _ in hits]


def version_of(d: date, versions: list[dict]) -> str | None:
    cur = None
    for v in sorted(versions, key=lambda v: v["start"]):
        if d >= date.fromisoformat(v["start"]):
            cur = v["name"]
    return cur


def doc_rules(cfg: dict, doc: str | None) -> tuple[str, list[dict], dict, dict]:
    """(문서 이름, 그 문서가 아는 판본 구간, 규칙 표 쪽, 항목 → 대조표 쪽). doc 이 없으면 가장 최근 문서."""
    # 문서를 안 주면 : 환경 변수 RAG_DOC (C의 eval_version.py를 고치지 않고 V3 정답지 채점 : RAG_DOC=V3) → 설정 default_doc(V2)
    doc = doc or os.environ.get("RAG_DOC") or cfg.get("default_doc") or list(cfg["docs"])[-1]
    d = cfg["docs"].get(doc, {})
    known = d.get("versions")
    versions = [v for v in cfg["versions"] if known is None or v["name"] in known]
    return doc, versions, d.get("rule_pages", cfg.get("rule_pages", {})), d.get("compare_pages", {})


def _bounds(versions: list[dict]) -> str:
    return ", ".join(f"{x['name']} {x['start']}~" for x in versions)


def _pick(question: str, dates: list[dict], acc: set[str], latest: bool = False) -> tuple[dict | None, str, list[dict]]:
    """기준일로 쓸 날짜 하나 (없으면 None), 설명, 후보.
    latest : 연속 야간근무처럼 '마지막 사건'이 판정을 가르는 항목 (설정 items[].pick = "latest")"""
    cand = [x for x in dates if x["roles"] & acc]
    if "가사" not in acc:   # '18시에 시작하는 설거지'의 시작은 근무 · 방문의 시작일이 아니다 (D067 · D138)
        cand = [x for x in cand if "가사" not in x["roles"]]
    if "추가주문" in acc and len(dates) > 1:   # 이미 체결한 주문은 빼고 이제 낼 주문의 날짜 (D097 · D100)
        fresh = [x for x in cand if "체결" not in x["roles"]]
        others = [x for x in dates if x not in cand and (not x["roles"] or x["roles"] & {"주문", "결제"})
                  and "체결" not in x["roles"]]
        if len(fresh) < len(cand) and (fresh or others):
            cand = fresh or others
    if not cand and "방문" in acc:   # 방문 날짜가 따로 없으면 '초대하려고'의 날짜가 방문일 (D079)
        cand = [x for x in dates if "초대" in x["roles"]]
    if NEW_TXN.search(question):   # "10월 5일에 전부 취소하고 새 주문번호로 다시 주문 확정" : 취소한 그날 새 거래
        for x in dates:
            after = re.split(r"[.!?\n]", question[x["end"]:])[0]
            nxt = [y["start"] for y in dates if y["start"] > x["start"]]
            after = after[: (nxt[0] - x["end"]) if nxt else None]
            if "취소" in x["roles"] and x not in cand and SAME_DAY_NEW.search(after):
                cand.append(x)
    if not cand:
        unroled = [x for x in dates if not x["roles"]]
        if not unroled:
            return None, "", []
        cand = unroled
    if len(cand) == 1:
        return cand[0], "", cand
    sep = [x for x in cand if SEPARATE.search(x.get("clause", ""))]
    if len(sep) == 1:   # '10-05에 1박을 따로 예약' : 별도 예약은 그 예약의 날짜 (D092)
        return sep[0], "별도 예약은 예약마다 판본을 정한다(제88조 · 제90조)", cand
    if NEW_TXN.search(question):
        # 취소 뒤에 나온 날짜가 새 일정이다 (날짜가 더 이르더라도, D086)
        last_cancel = max((m.start() for m in re.finditer("취소", question)), default=-1)
        after = [x for x in cand if x["start"] > last_cancel and "취소" not in x["roles"]]
        new = min(after, key=lambda x: x["start"]) if after else max(cand, key=lambda x: x["date"])
        return new, "취소 후 새 주문 · 예약 · 숙박은 새 날짜(부칙 제4조 · 제7조 · 제9조)", cand
    if latest:
        return max(cand, key=lambda x: x["date"]), "연속 근무는 판정 대상 당번 직전 근무의 시작일(부칙 제5조)", cand
    return min(cand, key=lambda x: x["date"]), "같은 사건의 날짜가 여럿이면 첫 확정일, 후속 결제도 같은 판본(제10조)", cand


def _decide_item(question: str, it: dict, dates: list[dict], versions: list[dict]) -> dict:
    acc = accepted_roles(it["basis"])
    res = {"item": it["name"], "basis": it["basis"], "basis_date": None, "applied_version": None}
    exact = [x for x in dates if not x.get("month")]
    months = [x for x in dates if x.get("month")]
    chosen, note, _ = _pick(question, exact, acc, latest=it.get("pick") == "latest")
    wd = it.get("weekdays")   # 요일이 정해진 항목 (V3 '금·토 술자리 귀가' : 일~목 시작 모임은 세 판본 공통, 표 55)
    if wd:
        named = ["월화수목금토일".index(x) for x in re.findall(r"([월화수목금토일])(?:요일|욜)", question)]   # '금욜' (S0779)
        days = [chosen["date"].weekday()] if chosen else named
        if days and not any(x in wd for x in days):
            res.update(status="개정 항목 아님",
                       reason=f"{it['basis']}이(가) {'·'.join('월화수목금토일'[x] for x in days)}요일 : "
                              f"{'·'.join('월화수목금토일'[x] for x in wd)}요일 시작이 아니라 판본마다 같은 규정(표 55). 판본 해당 없음")
            return res
    # 문의 · 고지 · 취소처럼 기준일이 아닌 역할만 붙은 달은 쓰지 않는다 ('9월에 알아봤다가 나중에 주문')
    months = [x for x in months if (x["roles"] & acc) or not (x["roles"] & {"문의", "고지", "취소", "환불"})]
    month_vers = {version_of(x["date"], versions) for x in months}
    if chosen is None and len(month_vers) > 1:   # '9월인지 10월인지 모르고' : 판본이 갈리는 달이 둘 이상
        res.update(status="추가 확인",
                   reason=f"질문의 달({', '.join(x['date'].strftime('%Y-%m') for x in months)})이 서로 다른 판본에 걸친다. "
                          f"'{it['basis']}'을(를) 묻는다(제8조 · 표 53)")
        return res
    if chosen is None and months:   # 일자 없이 달만 : 그달 전체가 한 판본 구간이면 그 판본
        m = min(months, key=lambda x: x["date"])
        first = m["date"]
        last = (first.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        v1, v2 = version_of(first, versions), version_of(last, versions)
        if v1 == v2:
            res.update(basis_date=first.strftime("%Y-%m"), applied_version=v1,
                       status="확정" if v1 else "규정 범위 밖",
                       reason=f"일자 없이 {first.year}년 {first.month}월 : 그달 전체가 {v1 or '시행 전'} 구간"
                              f"(시행 구간 {_bounds(versions)}). {it['basis']}의 일자는 판본을 바꾸지 않는다")
            return res
    if chosen is None:
        named = [n for n in NAMED_VERSION.findall(question) if n in {v["name"] for v in versions}]
        if len(set(named)) == 1:
            res.update(applied_version=named[0], status="확정",
                       reason=f"질문이 판본 {named[0]}을(를) 지정했다. {it['basis']}은(는) 질문에 없다")
            return res
    if chosen is None:
        if dates:
            seen = ", ".join(f"{x['date']}({'·'.join(sorted(x['roles'])) or '?'})" for x in dates)
            cond = f" 그 날짜로 시행 구간({_bounds(versions)})을 고른다." if versions else ""
            res.update(status="추가 확인",
                       reason=f"질문의 날짜 {seen}는 '{it['basis']}'이(가) 아니다. 그 날짜를 묻는다(제8조 · 표 53).{cond}")
        else:
            res.update(status="추가 확인", reason=f"'{it['basis']}'이(가) 질문에 없다. 판본이 갈리므로 그 날짜를 묻는다(제8조 · 표 53)")
        return res
    d = chosen["date"]
    v = version_of(d, versions)
    res["basis_date"] = d.isoformat()
    extras = []
    if note:
        extras.append(note)
    if any(x["roles"] & {"환불"} for x in dates):
        extras.append("환불은 원거래 판본 유지(제10조 · 제26조)")
    if any(x["roles"] & {"변경"} for x in dates) and not NEW_TXN.search(question):
        extras.append("색상 · 배송일만 바꾸면 원 주문일 유지(부칙 제4조)")
    if chosen["rel"]:
        extras.append(f"'{chosen['rel']}'을 앞 날짜 기준으로 계산")
    if v is None:
        res.update(status="규정 범위 밖", reason=f"{it['basis']} {d} 은 최초 시행일 전(표 53)")
    else:
        res.update(applied_version=v, status="확정",
                   reason=f"{it['basis']} {d} → {v} (시행 구간 {_bounds(versions)})" + "".join(f". {e}" for e in extras))
    return res


def decide(question: str, cfg: dict | None = None, doc: str | None = None) -> dict:
    """판본 판정. label = 적용_판본 칸에 쓸 팀 표기(V1 · V2 · 해당 없음 · 미정), label_raw = 규칙 원래 값
    ('추가 확인' · '규정 범위 밖' 포함), team_label = label 과 같음(11.19 이전 이름, 호환용). 규칙 버전 g3.3 (개발 문항과 다른 표현도 잡음 : 손님 · 오다 · 있다 가다 · 시간대 · 밤 근무 · 패키지, 열린 고지 · 합의면 비용 질문도 그 항목, 띄어쓰기 무시 키워드, V3 새 항목은 판정을 가르는 수치로 잡음, V2 판정은 g2.2와 같음)."""
    cfg = cfg or load_config()
    res = _decide(question, cfg, doc)
    raw = res.get("label") or res["applied_version"] or res["status"]
    res["label_raw"] = raw
    res["label"] = res["team_label"] = cfg.get("label_map", {}).get(raw, raw)
    res["rule_version"] = "g3.3"
    return res


def _decide(question: str, cfg: dict, doc: str | None) -> dict:
    doc, versions, pages, compare = doc_rules(cfg, doc)
    # 그 문서의 표 07에 있는 항목만 쓴다 (compare_pages 가 없으면 전부)
    active = [it for it in cfg["items"] if not compare or it["name"] in compare]
    found = detect_items(question, active)
    dates = find_dates(question, cfg.get("default_year", date.today().year),
                       cfg["docs"].get(doc, {}).get("this_month"))
    tag_roles(question, dates)
    res = {"doc": doc, "item": None, "basis": None, "candidates": [it["name"] for it, _ in found],
           "dates": [{"date": x["date"].isoformat(), "roles": sorted(x["roles"]), "text": x["clause"]} for x in dates],
           "basis_date": None, "applied_version": None, "status": None, "reason": "", "pages": [], "by_item": []}
    # label : 답변의 '적용 판본' 칸에 쓸 말 (V1 · V2 · 해당 없음 · 추가 확인 · 규정 범위 밖)

    if not found:   # 개정 항목이 아님 : 판본 '해당 없음'(팀 정답 표기). 되묻지 않는다
        chosen, _, _ = _pick(question, dates, GENERAL_ROLES)
        v = version_of(chosen["date"], versions) if chosen else None
        if chosen and v is None:
            res.update(basis_date=chosen["date"].isoformat(), status="규정 범위 밖",
                       reason=f"{chosen['date']} 은 최초 시행일 전(표 53)")
            return res
        cur = versions[-1]["name"] if versions else None
        res.update(basis_date=None, applied_version=None,
                   status="개정 항목 아님", label="해당 없음", text_version=cur,
                   reason=(f"사건일 {chosen['date']}. " if chosen else "") + f"표 07의 개정 항목이 아니다. 판본 해당 없음 — 판본마다 같은 규정이라 {cur} 본문 그대로 쓰고, 날짜를 되묻지 않는다")
        return res

    res["by_item"] = [_decide_item(question, it, dates, versions) for it, _ in found]
    main = res["by_item"][0]
    it = found[0][0]
    res.update({k: main[k] for k in ("item", "basis", "basis_date", "applied_version", "status", "reason")})
    res["label"] = main["applied_version"] or main["status"]
    pg = [pages.get("basis_table"), pages.get("effective_table")]
    for f, _ in found:
        pg.append(compare.get(f["name"], f.get("compare_page")))
    res["pages"] = list(dict.fromkeys(p for p in pg if p))
    if len(found) > 1:
        res["reason"] += " / 다른 항목 : " + "; ".join(
            f"{o['item']} {o['applied_version'] or o['status']}" for o in res["by_item"][1:])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--doc", default=None, help="문서 판본 (기본 : 설정의 마지막 문서)")
    a = ap.parse_args()
    print(json.dumps(decide(a.question, doc=a.doc), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
