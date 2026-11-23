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
RELATIVE = re.compile(r"(전날|전일|하루\s*전|다음\s*날|이튿날|익일|같은\s*날)" + _TIME)
REL_DAYS = {"전날": -1, "전일": -1, "하루전": -1, "다음날": 1, "이튿날": 1, "익일": 1, "같은날": 0}

# 날짜 뒤 구절을 끊는 곳 (한국어는 동사가 구절 끝에 온다)
CLAUSE_END = re.compile(r"[.,!?\n]|고\s|고$|는데|지만|다가|면서|며\s|서\s|라\s|니까|인데")

ROLE_WORDS = [  # (역할, 정규식). 한 구절에 여러 역할이 붙을 수 있다
    ("추가주문", r"추가\s*매수|추가\s*주문|물타기"),
    ("결제", r"결제|계산|지불|카드로|구입|구매|샀|사서|냈|썼|잔금|예약금"),
    ("주문", r"주문"),
    ("확정", r"확정"),
    ("예약", r"예약|예매|부킹"),
    ("시작", r"^\s*(?:에\s*)?부터|시작|출발|입실|체크인|출근"),
    ("숙박", r"숙박|묵|자고|재워|\d+\s*박"),
    ("방문", r"방문|놀러|와서|오기로|온다|오셔|오시|초대"),
    ("가사", r"배출|당번|청소|설거지|빨래"),
    ("종료", r"끝|종료|퇴근|체크아웃|퇴실"),
    ("고지", r"알렸|알려|고지|공지|말했|메시지|연락"),
    ("환불", r"환불|환급"),
    ("취소", r"취소"),
    ("문의", r"문의|알아보|장바구니|견적|상담|비교"),
    ("행사", r"생신|생일|명절|결혼식|장례|기일|행사"),
    ("배송", r"배송|도착|받았|받는"),
    ("변경", r"바꿨|바꾸|변경"),
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
    return acc or {"결제", "확정", "주문", "예약", "시작"}


SAME_DAY_NEW = re.compile(r"취소[^.]{0,30}(새|다시|재|별도)[^.]{0,20}(주문|예약|확정|잡)")
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


def find_dates(text: str, default_year: int) -> list[dict]:
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
        x.update(roles=roles, clause=clause.strip())


def detect_items(text: str, items: list[dict]) -> list[tuple[dict, int]]:
    """[(항목, 맞은 키워드 수)] 많은 순, 같으면 설정 순."""
    hits = []
    for n, it in enumerate(items):
        if any(w in text for w in it.get("none", [])):   # 바뀌지 않은 하위 규정을 묻는 말
            continue
        if it.get("any"):
            k = sum(1 for w in it["any"] if w in text)
            if k:
                hits.append((it, k, n))
        elif it.get("all"):
            ks = [sum(1 for w in g if w in text) for g in it["all"]]
            if all(ks):
                hits.append((it, sum(ks), n))
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
    doc = doc or list(cfg["docs"])[-1]
    d = cfg["docs"].get(doc, {})
    known = d.get("versions")
    versions = [v for v in cfg["versions"] if known is None or v["name"] in known]
    return doc, versions, d.get("rule_pages", cfg.get("rule_pages", {})), d.get("compare_pages", {})


def _bounds(versions: list[dict]) -> str:
    return ", ".join(f"{x['name']} {x['start']}~" for x in versions)


def _pick(question: str, dates: list[dict], acc: set[str]) -> tuple[dict | None, str, list[dict]]:
    """기준일로 쓸 날짜 하나 (없으면 None), 설명, 후보."""
    cand = [x for x in dates if x["roles"] & acc]
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
    if NEW_TXN.search(question):
        return max(cand, key=lambda x: x["date"]), "취소 후 새 주문 · 예약 · 숙박은 새 날짜(부칙 제4조 · 제7조 · 제9조)", cand
    return min(cand, key=lambda x: x["date"]), "같은 사건의 날짜가 여럿이면 첫 확정일, 후속 결제도 같은 판본(제10조)", cand


def _decide_item(question: str, it: dict, dates: list[dict], versions: list[dict]) -> dict:
    acc = accepted_roles(it["basis"])
    res = {"item": it["name"], "basis": it["basis"], "basis_date": None, "applied_version": None}
    chosen, note, _ = _pick(question, dates, acc)
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
    cfg = cfg or load_config()
    doc, versions, pages, compare = doc_rules(cfg, doc)
    # 그 문서의 표 07에 있는 항목만 쓴다 (compare_pages 가 없으면 전부)
    active = [it for it in cfg["items"] if not compare or it["name"] in compare]
    found = detect_items(question, active)
    dates = find_dates(question, cfg.get("default_year", date.today().year))
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
