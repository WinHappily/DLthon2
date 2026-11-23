"""판정 메모 : 모델이 '결론 + 이유(수치 비교 · 오해 반박) + 근거'를 쓰도록 검색 결과에서 사실만 뽑아 준다.

    python -m rag.brief "<질문>" --doc V2

판단은 하지 않는다(그건 모델 몫). 뽑는 것 :
  format    : 질문이 요구한 답 형식 (예/아니오 · 선택지 · 숫자 · 길이 · 한 문장)
  amounts   : 질문 속 금액과 합계 (합산할지는 제23조로 모델이 판단)
  version   : 적용 판본과 그 판본의 대조표 값 (표 54~57에서 맞은 행)
  rows      : 상위 쪽에서 질문과 맞은 표 행 (열 이름: 값)
  sentences : 상위 쪽 조항에서 질문 낱말과 가장 많이 겹치는 문장 (쪽 · 조항 표시)
"""
from __future__ import annotations

import argparse
import math
import re

from .common import tokenize

AMOUNT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*만\s*원")
FORMATS = [
    ("예/아니오", re.compile(r"예\s*/\s*아니오|예,?\s*아니오|예 또는 아니오|맞아\s*아니야")),
    ("선택지", re.compile(r"(?:^|\s)A[\s.)]*\S.*\sB[\s.)]*\S.*\sC[\s.)]")),
    ("숫자", re.compile(r"숫자만|몇\s*만\s*원|얼마(?:야|인가|인지|예요|에요)?\s*\??$|몇\s*(?:주|시간|박|회|명|%)")),
    ("한 문장", re.compile(r"한\s*문장")),
    ("상태만", re.compile(r"상태만|상태\s*하나만|한\s*마디로")),   # '가능 / 사전 합의 필요 / … 중 하나' (형식변형 F23~F26)
]
STATES = ["가능", "사전 합의 필요", "공동 처리 불가", "추가 확인", "규정에 없음"]
LENGTH = re.compile(r"(\d+)\s*[~∼-]\s*(\d+)\s*자")
SENT_SPLIT = re.compile(r"(?<=다\.)\s*|(?<=[.?!])\s+")


def detect_format(q: str) -> dict:
    out: dict = {"type": "서술"}
    for name, rx in FORMATS:
        if rx.search(q):
            out["type"] = name
            break
    if out["type"] == "선택지":
        out["options"] = sorted(set(re.findall(r"(?:^|[\s/])([A-E])(?=[\s.)가-힣])", q)))
    if out["type"] == "상태만":
        out["options"] = [x for x in STATES if x in q] or STATES
    m = LENGTH.search(q)
    if m:   # 글자 수를 정한 질문은 서술형 ('숫자만 채팅에 쓰고 …' 같은 문맥 오탐 방지)
        out = {"type": "서술", "length": f"{m.group(1)}~{m.group(2)}자"}
    if re.search(r"근거와 다음 행동|다음 행동", q):
        out["next_action"] = True
    return out


def _num(s: str) -> float:
    return float(s.replace(",", ""))


# 지출이 아니라 '기준 · 한도 · 잔고'인 금액 : 합계에서 뺀다
NOT_SPEND = re.compile(r"합의|승인|예산|잔액|잔고|가용|적립|상한|한도|기준|총액|계좌|월급|실수령|보유|평가액|자금|확보|여유")
DISCOUNT = re.compile(r"할인|포인트|캐시백|환급|환불|보전|쿠폰|돌려")


def amounts(q: str) -> dict:
    items = []
    for m in AMOUNT.finditer(q):
        ctx = re.split(r"[.,!?\n]|고\s|며\s|과\s|와\s", q[max(0, m.start() - 16):m.start()])[-1].strip()
        ctx = re.sub(r"^[\d\s년월일시분.:-]*(?:\d+일)?", "", ctx).strip()    # 앞에 붙은 날짜 조각
        ctx = re.sub(r"^(?:에서|에는|에|은|는|이|가|을|를)\s+", "", ctx)                 # 앞 조사
        kind = "기준" if NOT_SPEND.search(ctx) else "할인" if DISCOUNT.search(ctx) else "지출"
        items.append({"value": _num(m.group(1)), "context": ctx, "kind": kind})
    out = {"items": items, "values": [x["value"] for x in items]}
    spend = [x["value"] for x in items if x["kind"] == "지출"]
    if len(spend) >= 2:
        out["sum"] = round(sum(spend), 2)
    return out


def _fmt(x: float) -> str:
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def _articles(page_text: str) -> list[tuple[str, str]]:
    """[(조항, 본문)] '## 제N조 제목' 아래 문단."""
    out, cur, buf = [], None, []
    for line in page_text.splitlines():
        m = re.match(r"^## ((?:부칙 )?제\d+조)", line)
        if m or line.startswith("## 표") or line.startswith("# "):
            if cur and buf:
                out.append((cur, " ".join(buf)))
            cur, buf = (m.group(1) if m else None), []
            continue
        if cur and line.strip():
            buf.append(line.strip())
    if cur and buf:
        out.append((cur, " ".join(buf)))
    return out


def key_sentences(searcher, query: str, pages: list[int], n: int = 6) -> list[dict]:
    qt = set(tokenize(query))
    # 문서 안 흔한 토큰은 덜 센다 (BM25 의 idf 를 그대로 빌림)
    idf = getattr(searcher.bm25, "idf", {})
    scored = []
    for p in pages:
        for art, body in _articles(searcher.pages[p]["text"]):
            for s in SENT_SPLIT.split(body):
                s = s.strip()
                if len(s) < 8:
                    continue
                st = set(tokenize(s))
                sc = sum(idf.get(t, 1.0) for t in qt & st) / math.sqrt(len(st) + 1)
                if sc > 0:
                    scored.append((sc, p, art, s))
    scored.sort(key=lambda x: -x[0])
    out, seen = [], set()
    for sc, p, art, s in scored:
        if s in seen:
            continue
        seen.add(s)
        out.append({"page": p, "article": art, "text": s, "score": round(sc, 3)})
        if len(out) >= n:
            break
    return out


def _row_texts(searcher) -> dict[tuple[int, str], str]:
    if not hasattr(searcher, "_rowtext"):
        searcher._rowtext = {}
        for c in searcher.chunks:
            if c["kind"] == "row":
                searcher._rowtext[(c["page"], f"{c['table']} · {c['row']}")] = c["text"]
    return searcher._rowtext


def build_brief(searcher, res: dict, top: int = 5) -> dict:
    q = res.get("query") or res["question"]
    ver = res["version"]
    hits = res["hits"]
    rowtext = _row_texts(searcher)
    compare = set(searcher.cfg["docs"].get(searcher.doc, {}).get("compare_pages", {}).values())
    applied = ver.get("applied_version")

    rows, vrows = [], []
    for h in hits[:top] + [h for h in hits if h["page"] in compare]:
        for label in h["rows"]:
            t = rowtext.get((h["page"], label))
            if not t or t in rows or t in vrows:
                continue
            if h["page"] in compare:
                if applied and f"{applied}:" in t:
                    vrows.append(t)
            else:
                rows.append(t)
    pages = [h["page"] for h in hits[:top]]
    return {
        "format": detect_format(q),   # 정리된 질문 : 잡담 문단 속 '한 문장' 같은 오탐을 피한다
        "amounts": amounts(q),
        "version": {"label": ver.get("label"), "label_raw": ver.get("label_raw"), "item": ver.get("item"), "basis_date": ver.get("basis_date"),
                    "reason": ver.get("reason"), "rows": vrows},
        "rows": rows[:6],
        "sentences": key_sentences(searcher, q, pages),
    }


def brief_text(b: dict) -> str:
    f = b["format"]
    sep = " / " if f["type"] == "상태만" else " "
    fmt = f["type"] + (f" ({sep.join(f['options'])})" if f.get("options") else "") + (f", {f['length']}" if f.get("length") else "")
    lines = ["[판정 메모] 검색기가 뽑은 사실이다. 맞는지 발췌로 확인하고 판단은 직접 한다.",
             f"- 답 형식 : {fmt}"]
    a = b["amounts"]
    if a["items"]:
        tag = {"지출": "", "기준": "(기준·잔액)", "할인": "(할인·환급 : 뺄지 제24조로)"}
        s = ", ".join((f"{x['context']} " if x["context"] else "") + _fmt(x["value"]) + "만 원" + tag[x["kind"]]
                      for x in a["items"])
        lines.append(f"- 질문 금액 : {s}")
        if "sum" in a:
            lines.append(f"- 지출 금액 합계 : {_fmt(a['sum'])}만 원 (할인 · 기준 금액 제외. 같은 사건으로 합산할지는 제23조 · 제24조로 판단)")
    v = b["version"]
    if v["label"]:
        lab = v["label"] if v.get("label_raw") in (None, v["label"]) else f"{v['label']}({v['label_raw']})"
        lines.append(f"- 판본 : {lab}" + (f" ({v['item']}, 기준일 {v['basis_date']})" if v["item"] else "") + f" — {v['reason']}")
    for t in v["rows"]:
        lines.append(f"- 판본 대조 : {t}")
    for t in b["rows"]:
        lines.append(f"- 표 행 : {t}")
    for s in b["sentences"]:
        lines.append(f"- {s['page']}쪽 {s['article']} : {s['text']}")
    return "\n".join(lines)


def main():
    from .search import Searcher
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--doc", default="V2")
    a = ap.parse_args()
    s = Searcher(a.doc)
    res = s.retrieve(a.question)
    print(brief_text(build_brief(s, res)))


if __name__ == "__main__":
    main()
