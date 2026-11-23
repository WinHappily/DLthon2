"""판본 규칙만 따로 채점한다 (검색 없이 rag.version.decide 만).

    python -m eval.eval_version                                   # eval/questions_version.jsonl, 문서 V2
    python -m eval.eval_version --doc V3 --questions eval/questions_version_v3.jsonl

질문 한 줄 : {"id", "group", "question", "expect": {"item", "version", "basis_date", "also": {항목: 판본}}, "why"}
  version     : "V1" | "V2" | ... | "추가 확인" | "규정 범위 밖" | "해당 없음"(개정 항목 아님, 팀 정답 표기)
                "현행"(그 문서의 최신 판본)
  item        : 주 항목 이름, 개정 항목이 아니면 null
  also        : 한 질문에 항목이 여럿일 때 다른 항목의 판본
채점 (문항마다 모두 맞아야 정답) : 항목 · 판본(상태) · 기준일 · also
결과 : 화면(그룹별 · 틀린 문항) + results/version_<doc>_<시각>.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from rag.common import ROOT, load_config, read_jsonl
from rag.version import decide, doc_rules


def got_version(res: dict) -> str | None:
    if res["status"] in ("추가 확인", "규정 범위 밖"):
        return res["status"]
    if res["status"] == "개정 항목 아님":
        return "해당 없음"
    return res["applied_version"]


def grade(q: dict, res: dict, current: str | None) -> list[str]:
    exp, errs = dict(q["expect"]), []
    if exp["version"] == "현행":            # 그 문서의 최신 판본 (V2 문서면 V2, V3 문서면 V3)
        exp["version"] = current
    if "item" in exp and exp["item"] != res["item"]:
        errs.append(f"항목 {res['item']} (정답 {exp['item']})")
    if got_version(res) != exp["version"]:
        errs.append(f"판본 {got_version(res)} (정답 {exp['version']})")
    if exp.get("basis_date") and res["basis_date"] != exp["basis_date"]:
        errs.append(f"기준일 {res['basis_date']} (정답 {exp['basis_date']})")
    for name, v in (exp.get("also") or {}).items():
        other = next((o for o in res.get("by_item", []) if o["item"] == name), None)
        gv = got_version(other) if other else None
        if gv != v:
            errs.append(f"{name} {gv} (정답 {v})")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="V2")
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions_version.jsonl"))
    a = ap.parse_args()
    cfg = load_config()
    qs = read_jsonl(Path(a.questions))
    rows, by_group = [], defaultdict(lambda: [0, 0])
    for q in qs:
        res = decide(q["question"], cfg, a.doc)
        errs = grade(q, res, doc_rules(cfg, a.doc)[1][-1]["name"])
        by_group[q.get("group", "-")][0] += not errs
        by_group[q.get("group", "-")][1] += 1
        rows.append({"id": q["id"], "group": q.get("group"), "ok": not errs, "errors": errs,
                     "got": {"item": res["item"], "version": got_version(res), "basis_date": res["basis_date"]},
                     "expect": q["expect"], "reason": res["reason"]})
    n_ok = sum(r["ok"] for r in rows)
    print(f"[판본 규칙 · {a.doc}] {n_ok}/{len(rows)} 정답")
    for g, (ok, n) in by_group.items():
        print(f"  {g:<14} {ok}/{n}")
    bad = [r for r in rows if not r["ok"]]
    if bad:
        print("\n틀린 문항")
        for r in bad:
            print(f"  {r['id']} [{r['group']}] " + " · ".join(r["errors"]))
    out = ROOT / "results" / f"version_{a.doc}_{Path(a.questions).stem}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
