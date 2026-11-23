"""A-7 : 새 판본 PDF(V3 등)를 설정에 넣는다. 먼저 rag.extract 로 뽑은 뒤 실행한다.

    python -m rag.extract --doc V3 --pdf pdf/rulebook_V3.pdf
    python -m scripts.add_version --doc V3 --pdf pdf/rulebook_V3.pdf            # 확인만 (설정 안 바꿈)
    python -m scripts.add_version --doc V3 --pdf pdf/rulebook_V3.pdf --write    # config/versions.json 에 반영

뽑은 데이터(data/<doc>/)에서 자동으로 찾는 것
  시행 구간    : 본문의 'V3는 2026년 12월 1일 00시부터' 같은 문장 (부칙 제1조)
  기준일 표    : 열이 '항목 · 기준일 · 대조표'인 표 (V2의 표 07) → 항목마다 기준일 종류와 대조표 쪽
  시행 구간 표 : 열이 '구간 · 적용'인 표 (V2의 표 53)
  구조 확인    : 쪽 수, 표 번호 = 쪽 번호인 쪽 비율, 머리말 · 꼬리말이 남은 쪽, 추출 경고
새 항목(표 07에 새로 생긴 줄)은 items 에 '항목 이름의 낱말'을 키워드로 추가한다 → 동의어를 손으로 더하면 더 잘 잡는다.
기준일 종류는 문구('결제' · '주문' · '예약' · '시작' ...)로 날짜 역할을 정하므로(rag/version.py accepted_roles) 새 항목도 코드 수정 없이 판정된다.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date

from rag.common import ROOT, load_config, read_jsonl

START_RE = re.compile(r"(V\d+)\s*[은는]?\s*(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일\s*(?:\d{1,2}\s*시)?\s*부터")


def find_starts(pages: list[dict]) -> dict[str, str]:
    out = {}
    for p in pages:
        for m in START_RE.finditer(p["text"].replace("\n", " ")):
            name, d = m.group(1), date(int(m.group(2)), int(m.group(3)), int(m.group(4))).isoformat()
            out.setdefault(name, d)
    return dict(sorted(out.items(), key=lambda x: int(x[0][1:])))


def find_table(rows: list[dict], need: list[str]) -> tuple[int | None, list[dict]]:
    """열 이름에 need 의 낱말이 모두 들어 있는 첫 표의 (쪽, 행 목록)."""
    by_table: dict[tuple[int, str], list[dict]] = {}
    for r in rows:
        by_table.setdefault((r["page"], r["table"]), []).append(r)
    for (page, _), rs in sorted(by_table.items()):
        cols = list(rs[0]["cells"])
        if all(any(w in c for c in cols) for w in need):
            return page, rs
    return None, []


def col(cells: dict, word: str) -> str:
    return next((v for k, v in cells.items() if word in k), "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True, help="새 판본 이름 (예 : V3)")
    ap.add_argument("--pdf", required=True, help="그 판본 PDF (설정에 경로로 적힘)")
    ap.add_argument("--data", help="뽑은 데이터 폴더 (기본 data/<doc>)")
    ap.add_argument("--start", help="자동으로 못 찾을 때 이 판본의 시행일 YYYY-MM-DD")
    ap.add_argument("--write", action="store_true", help="config/versions.json 에 반영")
    a = ap.parse_args()

    data = a.data or f"data/{a.doc}"
    ddir = ROOT / data
    if not (ddir / "pages.jsonl").exists():
        raise SystemExit(f"{ddir}/pages.jsonl 가 없습니다. 먼저 python -m rag.extract --doc {a.doc} --pdf {a.pdf}")
    pages, rows = read_jsonl(ddir / "pages.jsonl"), read_jsonl(ddir / "rows.jsonl")
    report = (ddir / "report.txt").read_text(encoding="utf-8").splitlines()
    cfg = load_config()
    warn = []

    # 1. 구조 확인 (V2와 같은 방식으로 다뤄도 되는가)
    same = sum(1 for p in pages if p["tables"] and int(p["tables"][0].split()[1]) == p["page"])
    with_t = sum(1 for p in pages if p["tables"])
    leftover = [p["page"] for p in pages if re.search(r"NHC-\d+\s*/|프라이빗 AI 해커톤.*\d+\s*/\s*\d+", p["text"])]
    bad = [l for l in report if "⚠" in l]
    print(f"[구조] 쪽 {len(pages)} · 표 행 {len(rows)} · 표 번호 = 쪽 번호 {same}/{with_t}쪽"
          f" · 머리말/꼬리말 남은 쪽 {leftover or '없음'} · 추출 경고 {len(bad)}건")
    if with_t and same < with_t:
        warn.append(f"표 번호 ≠ 쪽 번호인 쪽이 {with_t - same}개 → 쪽 = 조각 가정(A-3) 확인 필요")
    if leftover:
        warn.append("머리말 · 꼬리말 형식이 바뀜 → rag/extract.py 의 HEADER_RE · FOOTER_RE 수정")
    for l in bad:
        print("   ", l)

    # 2. 시행 구간
    starts = find_starts(pages)
    if a.start:
        starts[a.doc] = a.start       # --doc 이름이 판본 이름(V3)과 같다고 본다
    print(f"[시행 구간] {starts or '못 찾음'}")
    if not set(starts) - {v["name"] for v in cfg["versions"]}:
        warn.append(f"기존 판본 말고 새 시행일을 못 찾음 → 부칙 제1조를 읽고 --start YYYY-MM-DD")

    # 3. 기준일 표 (표 07) · 시행 구간 표 (표 53)
    basis_page, basis_rows = find_table(rows, ["항목", "기준일", "대조표"])
    eff_page, _ = find_table(rows, ["구간", "적용"])
    print(f"[규칙 표] 기준일 표 {basis_page}쪽 · 시행 구간 표 {eff_page}쪽")
    if basis_page is None or eff_page is None:
        warn.append("기준일 표나 시행 구간 표를 못 찾음 → report.txt 와 PDF 확인")

    known = {it["name"]: it for it in cfg["items"]}
    compare, new_items = {}, []
    print("[개정 항목] (표 07)")
    for r in basis_rows:
        name, basis, ref = col(r["cells"], "항목"), col(r["cells"], "기준일"), col(r["cells"], "대조표")
        m = re.search(r"\d+", ref)
        cp = int(m.group()) if m else None
        compare[name] = cp
        tag = ""
        if name not in known:
            tag = "  ← 새 항목 : config 의 items 에 키워드를 채울 것"
            words = [w for w in re.findall(r"[가-힣A-Za-z]{2,}", name) if w not in ("상한", "기준", "항목", "개정")]
            new_items.append({"name": name, "basis": basis, "any": words,
                              "_todo": f"{a.doc}에서 새로 생긴 항목. any 는 항목 이름에서 자동으로 뽑았다 → 질문에 나올 낱말(동의어)을 더한다"})
        elif known[name]["basis"] != basis:
            tag = f"  ← 기준일 바뀜 (기존 '{known[name]['basis']}')"
            warn.append(f"'{name}' 기준일 종류가 바뀜 → items 의 basis 확인")
        print(f"   {name:<16} 기준일 {basis:<12} 대조표 {cp}쪽{tag}")
    for name in known:
        if basis_rows and name not in compare:
            print(f"   {name:<16} ← {a.doc}의 표에 없음 (이 문서에서는 판본 규칙에서 빠짐)")

    entry = {"pdf": a.pdf, "data": data, "versions": list(starts) or None,
             "rule_pages": {"basis_table": basis_page, "effective_table": eff_page},
             "compare_pages": compare}
    print("\n[설정에 들어갈 내용]")
    print(json.dumps({"docs": {a.doc: entry}}, ensure_ascii=False, indent=1))

    print("\n[확인할 것]" if warn else "\n[확인할 것] 없음")
    for w in warn:
        print("  -", w)

    if not a.write:
        print("\n확인만 했습니다. 반영하려면 --write")
        return
    have = {v["name"]: v for v in cfg["versions"]}
    for name, d in starts.items():
        if name in have and have[name]["start"] != d:
            print(f"  ! {name} 시행일 {have[name]['start']} → {d} 로 바꿈")
        have[name] = {"name": name, "start": d}
    cfg["versions"] = sorted(have.values(), key=lambda v: v["start"])
    cfg["docs"][a.doc] = entry
    cfg["items"].extend(new_items)
    path = ROOT / "config" / "versions.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n-> {path} 반영 (판본 {[v['name'] for v in cfg['versions']]}, 문서 {list(cfg['docs'])}, 새 항목 {len(new_items)}개)")


if __name__ == "__main__":
    main()
