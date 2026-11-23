"""A-1 · A-2 · A-3 : 규약집 PDF → 쪽 단위 조각 + 표 행 문장.

    python -m rag.extract --doc V2
    python -m rag.extract --doc V2 --pdf "<다른 경로>.pdf"

결과 (data/<doc>/)
  pages.jsonl        쪽 하나 = 조각 하나. 판본 · 쪽 · 장 · 조항 · 표 · 참조 · 개정 여부 · 본문
  rows.jsonl         표의 각 행을 '표 41 가족행사별 공동비 상한 V2, 행사: 부모님 생신, 행사당 상한: 22만 원 …' 문장으로
  article_map.json   '제23조' / '부칙 제3조' / '표 41' → 쪽 번호
  report.txt         쪽마다 조항 · 표 개수와 경고 (눈으로 확인할 것)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pdfplumber

from .common import ROOT, load_config

HEADER_RE = re.compile(r"^신혼생활 운영위원회 NHC-\d+\s*/")
FOOTER_RE = re.compile(r"^D61~65 프라이빗 AI 해커톤.*\d+\s*/\s*\d+\s*$")
ART_HEAD_RE = re.compile(r"^(부칙\s*)?제(\d+)조\s+(.+)$")
TABLE_TITLE_RE = re.compile(r"^표\s*(\d+)\s+(.+)$")
CHAPTER_RE = re.compile(r"^(제\d+장 .+|부칙|부록|차례)$")
REF_ART_RE = re.compile(r"(부칙\s*)?제\s*(\d+)\s*조")
REF_TAB_RE = re.compile(r"표\s*(\d+)((?:\s*[~·]\s*\d+)*)")
HANGUL_OR_DIGIT = re.compile(r"[가-힣0-9]")
HANGUL = re.compile(r"[가-힣]")


def join_wrapped(s: str) -> str:
    """칸 · 문단 안의 줄바꿈 복원. '회\\n식' → '회식', '표\\n50' → '표 50'."""
    parts = [p.strip() for p in s.split("\n") if p.strip()]
    if not parts:
        return ""
    acc = parts[0]
    for p in parts[1:]:
        if HANGUL_OR_DIGIT.match(acc[-1]) and HANGUL.match(p[0]):
            acc += p
        else:
            acc += " " + p
    return acc


def find_refs(text: str) -> tuple[list[str], list[str]]:
    """본문 속 '제23조', '부칙 제3조', '표 39~40', '표 41·54' 참조를 뽑는다."""
    arts = []
    for m in REF_ART_RE.finditer(text):
        arts.append(("부칙 " if m.group(1) else "") + f"제{int(m.group(2))}조")
    tabs = []
    for m in REF_TAB_RE.finditer(text):
        nums = [int(m.group(1))]
        tail = m.group(2) or ""
        for sep, n in re.findall(r"([~·])\s*(\d+)", tail):
            n = int(n)
            if sep == "~":
                nums.extend(range(nums[-1] + 1, n + 1))
            else:
                nums.append(n)
        tabs.extend(f"표 {n}" for n in nums)
    return list(dict.fromkeys(arts)), list(dict.fromkeys(tabs))


def _inside(obj, boxes) -> bool:
    cx, cy = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
    return any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in boxes)


def extract_page(page, doc: str) -> tuple[dict, list[dict], list[str]]:
    warns = []
    tables = page.find_tables()
    boxes = [t.bbox for t in tables]
    body = page.filter(lambda o: not _inside(o, boxes)) if boxes else page
    lines = [l.strip() for l in (body.extract_text() or "").split("\n") if l.strip()]
    lines = [l for l in lines if not HEADER_RE.match(l) and not FOOTER_RE.match(l)]

    chapter = lines[0] if lines and CHAPTER_RE.match(lines[0]) else ""
    rest = lines[1:] if chapter else lines
    title = rest[0] if rest else ""
    rest = rest[1:]

    # 제목 줄(조항 · 표 제목)에서 문단을 끊고, 그 사이 줄은 이어 붙인다
    blocks, cur = [], []
    articles, art_titles, table_titles = [], {}, {}
    for l in rest:
        ma, mt = ART_HEAD_RE.match(l), TABLE_TITLE_RE.match(l)
        if ma or mt:
            if cur:
                blocks.append(join_wrapped("\n".join(cur)))
                cur = []
            if ma:
                key = ("부칙 " if ma.group(1) else "") + f"제{int(ma.group(2))}조"
                articles.append(key)
                art_titles[key] = ma.group(3).strip()
                blocks.append(f"## {key} {ma.group(3).strip()}")
            else:
                key = f"표 {int(mt.group(1))}"
                table_titles[key] = mt.group(2).strip()
                blocks.append(f"## {key} {mt.group(2).strip()}")
        else:
            cur.append(l)
    if cur:
        blocks.append(join_wrapped("\n".join(cur)))

    rows_out, table_text = [], []
    tab_keys = list(table_titles)
    for ti, t in enumerate(tables):
        raw = t.extract()
        if not raw or len(raw) < 2:
            warns.append(f"표 추출 실패(행 {len(raw or [])}개)")
            continue
        key = tab_keys[ti] if ti < len(tab_keys) else f"표 {page.page_number}"
        ttitle = table_titles.get(key, "")
        header = [join_wrapped(c or "") for c in raw[0]]
        table_text.append(f"[{key} {ttitle}] " + " | ".join(header))
        for ri, r in enumerate(raw[1:], 1):
            cells = [join_wrapped(c or "") for c in r]
            pairs = [f"{h}: {c}" for h, c in zip(header, cells) if c]
            sent = f"{key} {ttitle}, " + ", ".join(pairs)
            table_text.append("- " + " | ".join(cells))
            rows_out.append({"doc": doc, "page": page.page_number, "table": key, "table_title": ttitle,
                             "row_no": ri, "row": cells[0] if cells else "", "cells": dict(zip(header, cells)),
                             "text": sent})
    if table_titles and not tables:
        warns.append("표 제목은 있는데 표를 찾지 못함")

    text_parts = ([f"# {chapter} · {title}"] if chapter else [f"# {title}"]) + blocks
    # 표는 표 제목 블록 바로 뒤에 넣는다
    if table_text:
        out, inserted = [], False
        for b in text_parts:
            out.append(b)
            if not inserted and b.startswith("## 표 "):
                out.extend(table_text)
                inserted = True
        if not inserted:
            out.extend(table_text)
        text_parts = out
    text = "\n".join(text_parts)

    ref_arts, ref_tabs = find_refs(text)
    own = set(articles) | set(table_titles) | {f"표 {page.page_number}"}
    ref_arts = [a for a in ref_arts if a not in own]
    ref_tabs = [t for t in ref_tabs if t not in own]
    amended = (any(a.startswith("부칙") for a in articles + ref_arts)
               or any(re.search(r"\bV\d\b", t) for t in table_titles.values())
               or bool(re.search(r"\bV\d\b", text)))

    if not chapter and page.page_number > 1:
        warns.append("장 제목 못 찾음")
    rec = {"doc": doc, "page": page.page_number, "id": f"{doc}-p{page.page_number:02d}",
           "chapter": chapter, "title": title, "articles": articles, "article_titles": art_titles,
           "tables": list(table_titles), "table_titles": table_titles,
           "refs": {"articles": ref_arts, "tables": ref_tabs}, "amended": amended, "text": text}
    return rec, rows_out, warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True, help="판본 이름 (config/versions.json 의 docs 키)")
    ap.add_argument("--pdf", help="PDF 경로 (기본 : 설정 파일)")
    ap.add_argument("--out", help="출력 폴더 (기본 : 설정 파일)")
    args = ap.parse_args()
    cfg = load_config()
    dcfg = cfg["docs"].get(args.doc, {})
    pdf_path = Path(args.pdf or (ROOT / dcfg["pdf"]))
    out = Path(args.out or (ROOT / dcfg.get("data", f"data/{args.doc}")))
    out.mkdir(parents=True, exist_ok=True)

    pages, rows, report = [], [], []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            rec, rws, warns = extract_page(p, args.doc)
            pages.append(rec)
            rows.extend(rws)
            report.append(f"p{p.page_number:02d}  조항 {len(rec['articles'])}  표 {len(rec['tables'])}  행 {len(rws)}"
                          f"  개정 {'O' if rec['amended'] else '-'}  {rec['title']}" + (f"  ⚠ {'; '.join(warns)}" if warns else ""))

    amap = {}
    for rec in pages:
        for a in rec["articles"]:
            amap.setdefault(a, rec["page"])
        for t in rec["tables"]:
            amap.setdefault(t, rec["page"])
    with open(out / "pages.jsonl", "w", encoding="utf-8") as f:
        for r in pages:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "article_map.json").write_text(json.dumps(amap, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "report.txt").write_text("\n".join(report), encoding="utf-8")

    n_art = sum(1 for k in amap if "조" in k)
    bad = [l for l in report if "⚠" in l]
    print(f"[{args.doc}] 쪽 {len(pages)} · 표 행 {len(rows)} · 조항 {n_art} · 표 {len(amap) - n_art} -> {out}")
    print(f"경고 {len(bad)}건" + ("" if not bad else ":\n  " + "\n  ".join(bad)))


if __name__ == "__main__":
    main()
