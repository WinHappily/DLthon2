"""BM25만 vs BM25 + bge-m3 : 필수 근거가 검색에 들어오는 비율 (C 지시 3-5, bge-m3 채택 근거). 모델 실행 없음.

    python -m eval.eval_dense --dev <개발정답_150.csv>        # 코랩 : 2절 색인(emb.npy)을 만든 뒤

필수 근거(근거_채점_대상 = 예)의 ';' 묶음마다 '|' 후보 중 하나가 있는 쪽이 들어왔는가.
  recall@5    : 검색 상위 5쪽 (참조 · 판본 · 고정 쪽 제외)
  recall@기본 : retrieve() 결과 전체 (고정 쪽 제외 : fixed=False)
"""
from __future__ import annotations

import argparse
import csv

from eval.eval_noise import covered, need_groups
from rag.search import Searcher


def run(s: Searcher, rows: list[dict], miss: dict | None = None) -> tuple[int, int, int, float]:
    n = a5 = ad = hits = 0
    for r in rows:
        res = s.retrieve(r["질문"], fixed=False)
        hits += len(res["hits"])
        g = need_groups(r)
        if g is None:
            continue
        n += 1
        a5 += covered(g, {h["page"] for h in res["hits"][:5]})
        ok = covered(g, {h["page"] for h in res["hits"]})
        ad += ok
        if miss is not None and not ok:
            have = {h["page"] for h in res["hits"]}
            miss[r["문항번호"]] = ([sorted(x) for x in g if not x & have], [h["page"] for h in res["hits"]])
    return n, a5, ad, hits / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--doc", default="V2")
    ap.add_argument("--fmt", help="형식변형 csv : 이 문항들로 잰다. 쪽 번호 · 근거_채점_대상은 --dev의 원문항에서 가져온다")
    ap.add_argument("--show", action="store_true", help="방식별로 필수 근거가 빠진 문항 · 빠진 쪽 · 들어온 쪽 출력")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.dev, encoding="utf-8-sig")))
    if a.fmt:
        dev = {r["문항번호"]: r for r in rows}
        rows = [{**r, "근거_채점_대상": dev[r["원문항"]]["근거_채점_대상"], "보조_근거_쪽": dev[r["원문항"]]["보조_근거_쪽"]}
                for r in csv.DictReader(open(a.fmt, encoding="utf-8-sig"))]
    s = Searcher(a.doc)
    dense = s.dense
    print("검색 방식 | recall@5 | recall@기본 | 평균 hits 수")
    out, misses = [], {}
    if dense:
        misses["BM25 + bge-m3"] = {}
        out.append(("BM25 + bge-m3", run(s, rows, misses["BM25 + bge-m3"])))
    s.dense = None
    misses["BM25만"] = {}
    out.append(("BM25만", run(s, rows, misses["BM25만"])))
    s.dense = dense
    for name, (n, a5, ad, h) in out:
        print(f"{name} | {a5}/{n} ({a5 / n:.3f}) | {ad}/{n} ({ad / n:.3f}) | {h:.1f}")
    if a.show:
        for name, m in misses.items():
            print(f"\n[{name}] 필수 근거 빠진 문항 {len(m)}개")
            for qid, (lost, got) in m.items():
                print(f"  {qid} 빠진 쪽(후보) {lost} · 들어온 쪽 {got}")
    if not dense:
        print("(벡터 파일 없음 : 2절 색인을 --embed BAAI/bge-m3 로 만든 뒤 다시)")


if __name__ == "__main__":
    main()
