"""(A) PDF 추출 시험 : 본문 · 표 · 쪽 번호가 살아 있는지 몇 쪽으로 눈과 자동 검사로 확인한다.

    python -m scripts.check_extract --doc V2                  # 기본 : 17 · 41 · 54쪽
    python -m scripts.check_extract --doc V2 --pages 3 17 59

검사 항목 (쪽마다)
  쪽 번호      : 꼬리말 'NN / 60'의 NN과 조각의 page 가 같은가 (PDF에서 다시 읽어 대조)
  본문         : 조항 제목(제N조)이 1개 이상 있고 본문 글자가 50자 이상인가
  표           : 표 제목(표 NN)의 번호 = 쪽 번호인가, 표 행이 1개 이상 뽑혔는가
  줄바꿈 복원  : 표 칸 안에 줄바꿈(\\n)이 남아 있지 않은가
  알려진 행    : 17쪽이면 '같은 누수의 출동과 후속부품' 행이 한 행으로 있는가
결과 : 화면 + results/check_extract_<doc>_<시각>.txt
"""
from __future__ import annotations

import argparse
import re
import time

import pdfplumber

from rag.common import ROOT, load_config, read_jsonl

KNOWN_ROWS = {17: "같은 누수의 출동과 후속부품", 54: "전량 취소 후 새 주문", 59: "금요일 22시 시작한 감석 회식"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="V2")
    ap.add_argument("--pdf", help="쪽 번호 대조용 원본 PDF (기본 : 설정 파일)")
    ap.add_argument("--pages", type=int, nargs="+", default=[17, 41, 54])
    args = ap.parse_args()
    cfg = load_config()
    dcfg = cfg["docs"].get(args.doc, {})
    ddir = ROOT / dcfg.get("data", f"data/{args.doc}")
    pages = {p["page"]: p for p in read_jsonl(ddir / "pages.jsonl")}
    rows = read_jsonl(ddir / "rows.jsonl")
    pdf_path = args.pdf or (ROOT / dcfg["pdf"] if "pdf" in dcfg else None)

    footer_no = {}
    if pdf_path:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for n in args.pages:
                    last = (pdf.pages[n - 1].extract_text() or "").strip().split("\n")[-1]
                    m = re.search(r"(\d+)\s*/\s*\d+\s*$", last)
                    footer_no[n] = int(m.group(1)) if m else None
        except FileNotFoundError:
            print(f"(PDF {pdf_path} 를 못 찾아 쪽 번호 대조는 건너뜀. --pdf 로 지정)")

    out, fails = [], 0
    for n in args.pages:
        p = pages[n]
        prow = [r for r in rows if r["page"] == n]
        checks = []
        if n in footer_no:
            checks.append(("쪽 번호(꼬리말 대조)", footer_no[n] == n, f"꼬리말 {footer_no[n]} / 조각 {n}"))
        body = re.sub(r"^## .*$|^- .*$|^\[표.*$", "", p["text"], flags=re.M)
        checks.append(("본문 조항", len(p["articles"]) >= 1 or n in (1, 2, 58, 59, 60), ", ".join(p["articles"]) or "없음"))
        checks.append(("본문 길이", len(body.strip()) >= 50, f"{len(body.strip())}자"))
        tnums = [int(t.split()[1]) for t in p["tables"]]
        checks.append(("표 번호 = 쪽 번호", tnums == [n], ", ".join(p["tables"]) or "없음"))
        checks.append(("표 행 추출", len(prow) >= 1, f"{len(prow)}행"))
        bad = [r["row"] for r in prow if any("\n" in v for v in r["cells"].values())]
        checks.append(("칸 안 줄바꿈 복원", not bad, "남은 줄바꿈 " + str(bad) if bad else "모두 한 줄"))
        if n in KNOWN_ROWS:
            ok = any(r["row"].startswith(KNOWN_ROWS[n]) for r in prow)
            checks.append(("알려진 행", ok, KNOWN_ROWS[n]))

        out.append(f"\n{'=' * 70}\n{p['id']} · {p['chapter']} · {p['title']}\n{'=' * 70}")
        for name, ok, detail in checks:
            fails += not ok
            out.append(f"  [{'통과' if ok else '실패'}] {name:<14} {detail}")
        out.append("\n--- 조각 본문 (프롬프트에 들어가는 그대로) ---")
        out.append(p["text"])
        out.append("\n--- 표 행 문장 (검색용 보조 조각) ---")
        out.extend("  " + r["text"] for r in prow)

    out.append(f"\n실패 {fails}건 / 쪽 {len(args.pages)}개")
    text = "\n".join(out)
    print(text)
    res = ROOT / "results" / f"check_extract_{args.doc}_{time.strftime('%Y%m%d-%H%M')}.txt"
    res.parent.mkdir(exist_ok=True)
    res.write_text(text, encoding="utf-8")
    print(f"-> {res}")


if __name__ == "__main__":
    main()
