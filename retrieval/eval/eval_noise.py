"""잡음 변형(N 세트)이 원문항보다 검색 · 판본에서 나빠지는지 본다 (C 지시 3-3 · 3-1).

    python -m eval.eval_noise --noise <잡음문항.jsonl 또는 .csv> --dev <개발정답_150.csv> [--variant c]

- 원문항 대응 : csv에 원문항 칸이 있으면 그것, 없으면 '원문항 30개 × (a, b, c) 순서'로 보고 a변형과 가장 비슷한 개발 문항
- 근거 : 정답 csv의 필수_근거(';' = 모두, '|' = 하나)를 보조_근거_쪽의 '조항=쪽'으로 쪽 번호로 바꿔, 검색 결과(hits)에 들어왔는지
  근거_채점_대상 = 예 인 문항만. 고정 조각(4 · 6 · 7 · 53쪽)은 빼고 잰다(검색 자체 성능)
- 판본 : team_label 이 원문항과 같은지
결과 : 화면 (원문항보다 나빠진 문항, 대표 query 전/후)
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import unicodedata
from pathlib import Path

from rag.search import Searcher


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


def load_noise(path: Path, dev: dict[str, dict]) -> list[dict]:
    if path.suffix == ".csv":
        rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
        out = []
        for r in rows:
            orig = r.get("원문항") or r.get("원문항번호") or r.get("원_문항번호")
            var = r.get("변형", "")
            out.append({"id": r["문항번호"], "question": r["질문"], "orig": orig, "variant": var[:1].lower() or "?"})
        return out
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    out = []
    for i in range(0, len(rows), 3):                  # 원문항마다 a(표기) · b(압박) · c(잡담) 순서
        a = rows[i]
        orig = max(dev, key=lambda d: difflib.SequenceMatcher(None, _norm(dev[d]["질문"])[:200], _norm(a["질문"])[:300]).ratio())
        for r, v in zip(rows[i:i + 3], "abc"):
            out.append({"id": r["문항번호"], "question": r["질문"], "orig": orig, "variant": v})
    return out


def need_groups(r: dict) -> list[set[int]] | None:
    """필수_근거 → [{쪽 후보}] (';' 묶음마다 '|' 후보의 쪽). 쪽을 모르면 None."""
    if r.get("근거_채점_대상") != "예" or not r.get("필수_근거"):
        return None
    where = {k.strip(): int(v) for k, v in re.findall(r"([^;=]+)=(\d+)쪽", r.get("보조_근거_쪽", ""))}
    groups = []
    for g in r["필수_근거"].split(";"):
        pages = {where[x.strip()] for x in g.split("|") if x.strip() in where}
        if not pages:
            return None
        groups.append(pages)
    return groups


def covered(groups: list[set[int]], pages: set[int]) -> bool:
    return all(g & pages for g in groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--variant", default="c", help="볼 변형 (a / b / c / all)")
    ap.add_argument("--doc", default="V2")
    a = ap.parse_args()
    dev = {r["문항번호"]: r for r in csv.DictReader(open(a.dev, encoding="utf-8-sig"))}
    noise = [n for n in load_noise(Path(a.noise), dev) if a.variant == "all" or n["variant"] == a.variant]
    s = Searcher(a.doc)
    worse, n_eval, ok_orig, ok_var, ver_diff, samples = [], 0, 0, 0, [], []
    for n in noise:
        o = dev[n["orig"]]
        ro = s.retrieve(o["질문"], fixed=False)
        rn = s.retrieve(n["question"], fixed=False)
        if ro["version"]["team_label"] != rn["version"]["team_label"]:
            ver_diff.append(f"{n['id']}({n['orig']}) {ro['version']['team_label']} → {rn['version']['team_label']}")
        g = need_groups(o)
        if g is None:
            continue
        n_eval += 1
        po, pn = {h["page"] for h in ro["hits"]}, {h["page"] for h in rn["hits"]}
        co, cn = covered(g, po), covered(g, pn)
        ok_orig += co
        ok_var += cn
        if co and not cn:
            worse.append(f"{n['id']}({n['orig']}) 빠진 쪽 {sorted(set().union(*g) - pn)}")
        if len(samples) < 3 and len(n["question"]) > 400:
            samples.append((n["id"], len(n["question"]), rn["query"]))
    print(f"[잡음 {a.variant} · {a.doc}] {len(noise)}문항 (근거 채점 {n_eval}) · 고정 조각 제외")
    print(f"  필수 근거가 검색에 다 들어온 문항 : 원문항 {ok_orig}/{n_eval} → 잡음 {ok_var}/{n_eval}")
    print(f"  원문항보다 나빠진 문항 {len(worse)}개" + ("".join(f"\n    {w}" for w in worse)))
    print(f"  판본이 원문항과 다른 문항 {len(ver_diff)}개" + ("".join(f"\n    {w}" for w in ver_diff)))
    for i, L, q in samples:
        print(f"  [query 예] {i} ({L}자) → {q[:160]!r}")


if __name__ == "__main__":
    main()
