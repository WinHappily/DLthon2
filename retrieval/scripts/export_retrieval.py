"""질문 세트 전체의 검색 결과 · 판정 메모 · 프롬프트를 파일로 만든다 (B · C에게 넘기는 산출물).

    python -m scripts.export_retrieval --questions <파일> --name dev50          # 코랩에서 bge-m3 켜고 실행 권장
    python -m scripts.export_retrieval --questions <키트>/01_공개_배포평가셋_100문항.jsonl --name public100

질문 파일 : jsonl ({"id"|"문항번호", "question"|"질문"}) 또는 csv (문항번호, 질문)
결과 : results/retrieval_<name>_<doc>.jsonl  한 줄 = 한 질문
  id · question(원문) · query(검색용 정리) · version · brief(판정 메모 dict) · brief_text
  hits[page, title, articles, tables, rows, why, text] · messages(그대로 모델에 넣는 대화) · prompt_chars
B는 messages 를 vLLM chat 에 그대로 넣으면 된다. 같은 파일을 모든 모델 실험(E0~E3)에 재사용하면
검색이 고정되어 '모델 차이'만 비교된다.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from rag.brief import brief_text, build_brief
from rag.common import ROOT, read_jsonl
from rag.prompt import build_messages
from rag.search import Searcher


def load_questions(path: Path) -> list[dict]:
    if path.suffix == ".csv":
        rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    else:
        rows = read_jsonl(path)
    return [{"id": r.get("id") or r.get("문항번호"), "question": r.get("question") or r.get("질문")} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--doc", default="V2")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--max-refs", type=int, default=4)
    ap.add_argument("--no-brief", action="store_true", help="판정 메모 없이 (비교 실험용)")
    ap.add_argument("--no-fixed", action="store_true", help="고정 조각(4 · 6 · 7 · 53쪽) 없이")
    ap.add_argument("--no-cases", action="store_true", help="공개 사례 쪽(표 58 · 59)을 검색에서 뺀다")
    a = ap.parse_args()

    qs = load_questions(Path(a.questions))
    s = Searcher(a.doc)
    out = ROOT / "results" / f"retrieval_{a.name}_{a.doc}.jsonl"
    out.parent.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    chars = []
    with open(out, "w", encoding="utf-8") as f:
        for q in qs:
            t = time.perf_counter()
            res = s.retrieve(q["question"], k=a.k, max_refs=a.max_refs, fixed=not a.no_fixed, cases=not a.no_cases)
            b = build_brief(s, res)
            bt = None if a.no_brief else brief_text(b)
            msgs = build_messages(q["question"], res, bt)
            n = sum(len(m["content"]) for m in msgs)
            chars.append(n)
            f.write(json.dumps({
                "id": q["id"], "doc": a.doc, "question": q["question"], "query": res["query"],
                "version": res["version"], "brief": b, "brief_text": bt,
                "hits": [{k: h[k] for k in ("page", "title", "articles", "tables", "rows", "why", "text")} for h in res["hits"]],
                "messages": msgs, "prompt_chars": n, "retrieve_sec": round(time.perf_counter() - t, 3),
            }, ensure_ascii=False) + "\n")
    print(f"[{a.name} · {a.doc}] {len(qs)}문항 {time.perf_counter() - t0:.1f}초, 벡터 검색 {'켬' if s.dense else '끔(BM25만)'}")
    print(f"  프롬프트 평균 {sum(chars) / len(chars):,.0f}자 (최대 {max(chars):,}자)  -> {out}")


if __name__ == "__main__":
    main()
