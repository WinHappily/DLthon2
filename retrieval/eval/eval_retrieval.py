"""A-5 : 검색 품질 측정. 정답 근거(조항 · 표)가 있는 쪽이 검색 결과에 들어오는지 잰다.

    python -m eval.eval_retrieval --doc V2
    python -m eval.eval_retrieval --doc V2 --questions eval/questions_team.jsonl -k 5

질문 파일 한 줄 : {"id", "question", "expected_refs": ["제23조", "표 41", "부칙 제3조"], "expected_version": "V1"|"V2"|"추가 확인"|"무관"|null}
  expected_version 은 문서마다 다르면 {"V2": "V2", "V3": "V3"} 처럼 써도 된다
지표
  page_recall  정답 쪽 중 결과에 들어온 비율 (문항 평균)
  all_hit      정답 쪽이 모두 들어온 문항 비율 (근거 인용 정확도의 상한)
  mrr          첫 정답 쪽의 순위 역수
  version_acc  판본 규칙이 맞은 비율 (expected_version 이 null 이면 제외)
결과 : results/retrieval_<doc>_<날짜>.jsonl (문항별) + 요약 출력
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rag.common import ROOT, read_jsonl
from rag.search import Searcher


def version_ok(expected, ver: dict) -> bool | None:
    if isinstance(expected, dict):          # {"V2": "V2", "V3": "V3"} : 문서마다 정답이 다를 때
        expected = expected.get(ver.get("doc"), expected.get("default"))
    if expected is None:
        return None
    if expected == "무관":
        return ver["status"] == "개정 항목 아님"
    if expected == "추가 확인":
        return ver["status"] == "추가 확인"
    return ver["applied_version"] == expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="V2")
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions_seed.jsonl"))
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--max-refs", type=int, default=4)
    args = ap.parse_args()

    s = Searcher(args.doc)
    qs = read_jsonl(Path(args.questions))
    rows = []
    for mode, follow in (("검색만", False), ("검색+참조+판본", True)):
        rec, allh, mrr, vok, vn = 0.0, 0, 0.0, 0, 0
        for q in qs:
            want = sorted({s.amap[r] for r in q["expected_refs"] if r in s.amap})
            missing_ref = [r for r in q["expected_refs"] if r not in s.amap]
            if follow:
                res = s.retrieve(q["question"], k=args.k, max_refs=args.max_refs)
                hits, ver = res["hits"], res["version"]
            else:
                hits, ver = s.search(q["question"], k=args.k, follow_refs=False), None
            got = [h["page"] for h in hits]
            inter = [p for p in want if p in got]
            r = len(inter) / len(want) if want else 0
            first = next((i for i, p in enumerate(got) if p in want), None)
            rec += r
            allh += int(len(inter) == len(want))
            mrr += 1 / (first + 1) if first is not None else 0
            row = {"mode": mode, "id": q["id"], "want": want, "got": got, "missed": [p for p in want if p not in got],
                   "recall": round(r, 3), "missing_ref": missing_ref}
            if ver is not None:
                ok = version_ok(q.get("expected_version"), ver)
                row.update(version=ver["applied_version"] or ver["status"], version_expected=q.get("expected_version"),
                           version_ok=ok)
                if ok is not None:
                    vn += 1
                    vok += int(ok)
            rows.append(row)
        n = len(qs)
        line = f"[{mode:<12}] k={args.k}  page_recall {rec / n:.3f}  all_hit {allh}/{n}  mrr {mrr / n:.3f}"
        if follow:
            line += f"  version_acc {vok}/{vn}  (평균 쪽 수 {sum(len(r['got']) for r in rows if r['mode'] == mode) / n:.1f})"
        print(line)

    print("\n놓친 쪽 (검색+참조+판본)")
    for r in rows:
        if r["mode"] == "검색+참조+판본" and (r["missed"] or r.get("version_ok") is False):
            v = f"  판본 {r['version']} (정답 {r['version_expected']})" if r.get("version_ok") is False else ""
            print(f"  {r['id']:<7} 놓친 쪽 {r['missed']}{v}")

    out = ROOT / "results" / f"retrieval_{args.doc}_{time.strftime('%Y%m%d-%H%M')}.jsonl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
