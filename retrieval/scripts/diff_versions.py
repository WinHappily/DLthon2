"""A-7 : 두 판본 문서의 차이 (조항 · 표 행) + 같은 질문의 판정 차이.

    python -m scripts.diff_versions --old V2 --new V3
    python -m scripts.diff_versions --old V2 --new V3 --questions eval/questions_seed.jsonl

비교 단위
  조항   : '제23조' · '부칙 제3조' 번호로 맞춘다. 본문 글자가 다르면 '바뀜'(바뀐 부분만 보여 줌)
  표 행  : 표 '제목'과 행 첫 칸으로 맞춘다(쪽이 밀려 표 번호가 바뀌어도 따라감). 칸 값이 다르면 '바뀜'
  질문   : 같은 질문을 두 문서로 retrieve → 적용 판본 · 상태 · 상위 쪽이 달라진 문항
결과 : results/diff_<old>_<new>.md  (C에게 넘길 '개정 대응 질문' 재료)
"""
from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path

from rag.common import ROOT, load_config, read_jsonl

ART_SPLIT = re.compile(r"^## ((?:부칙 )?제\d+조) (.*)$", re.M)


def articles(pages: list[dict]) -> dict[str, dict]:
    """조항 번호 → {쪽, 제목, 본문}. 본문은 다음 '## ' 머리까지."""
    out = {}
    for p in pages:
        t = p["text"]
        for m in ART_SPLIT.finditer(t):
            end = t.find("\n## ", m.end())
            body = t[m.end():end if end != -1 else len(t)].strip()
            out[m.group(1)] = {"page": p["page"], "title": m.group(2).strip(), "body": body}
    return out


def rows_by_key(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["table_title"], r["row"]): r for r in rows}


def short_diff(a: str, b: str, width: int = 40) -> str:
    """바뀐 부분만 '…앞[-지운 것-]{+넣은 것+}뒤…' 로."""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    parts = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        ctx = a[max(0, i1 - 12):i1]
        parts.append(f"…{ctx}" + (f"[-{a[i1:i2]}-]" if i2 > i1 else "") + (f"{{+{b[j1:j2]}+}}" if j2 > j1 else "")
                     + f"{a[i2:i2 + 12]}…")
    return "  ".join(parts[:4]) + (f"  (외 {len(parts) - 4}곳)" if len(parts) > 4 else "")


def load(doc: str, cfg: dict):
    d = ROOT / cfg["docs"].get(doc, {}).get("data", f"data/{doc}")
    return read_jsonl(d / "pages.jsonl"), read_jsonl(d / "rows.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="V2")
    ap.add_argument("--new", default="V3")
    ap.add_argument("--questions", help="같은 질문을 두 문서로 돌려 판정 차이도 본다 (jsonl)")
    a = ap.parse_args()
    cfg = load_config()
    op, orows = load(a.old, cfg)
    np_, nrows = load(a.new, cfg)
    out = [f"# {a.old} → {a.new} 차이", "",
           f"- 쪽 수 {len(op)} → {len(np_)}, 표 행 {len(orows)} → {len(nrows)}"]

    # 조항
    oa, na = articles(op), articles(np_)
    added = [k for k in na if k not in oa]
    removed = [k for k in oa if k not in na]
    changed = [k for k in na if k in oa and (oa[k]["body"] != na[k]["body"] or oa[k]["title"] != na[k]["title"])]
    moved = [k for k in na if k in oa and oa[k]["page"] != na[k]["page"]]
    out += [f"- 조항 : 새로 생김 {len(added)} · 없어짐 {len(removed)} · 바뀜 {len(changed)} · 쪽 이동 {len(moved)}", ""]
    out.append("## 조항\n")
    out.append("| 조항 | 상태 | 쪽 (old→new) | 제목 | 바뀐 부분 |")
    out.append("|---|---|---|---|---|")
    for k in added:
        out.append(f"| {k} | 새로 생김 | – → {na[k]['page']} | {na[k]['title']} | {na[k]['body'][:80]}… |")
    for k in removed:
        out.append(f"| {k} | 없어짐 | {oa[k]['page']} → – | {oa[k]['title']} | |")
    for k in changed:
        out.append(f"| {k} | 바뀜 | {oa[k]['page']} → {na[k]['page']} | {na[k]['title']} | "
                   f"{short_diff(oa[k]['title'] + ' ' + oa[k]['body'], na[k]['title'] + ' ' + na[k]['body'])} |")
    if moved:
        out.append(f"\n쪽만 이동 : " + ", ".join(f"{k} {oa[k]['page']}→{na[k]['page']}" for k in moved[:30])
                   + (" …" if len(moved) > 30 else ""))

    # 표 행
    orb, nrb = rows_by_key(orows), rows_by_key(nrows)
    radd = [k for k in nrb if k not in orb]
    rdel = [k for k in orb if k not in nrb]
    rchg = [k for k in nrb if k in orb and orb[k]["cells"] != nrb[k]["cells"]]
    out += ["", "## 표 행\n", f"새 행 {len(radd)} · 없어진 행 {len(rdel)} · 값이 바뀐 행 {len(rchg)}", "",
            "| 표 (old→new) | 행 | 상태 | 바뀐 칸 |", "|---|---|---|---|"]
    for k in rchg:
        o, n = orb[k], nrb[k]
        diffs = [f"{c}: {o['cells'].get(c, '–')} → {n['cells'].get(c, '–')}"
                 for c in dict.fromkeys(list(o["cells"]) + list(n["cells"])) if o["cells"].get(c) != n["cells"].get(c)]
        out.append(f"| {o['table']}→{n['table']} {k[0]} | {k[1]} | 바뀜 | {'; '.join(diffs)} |")
    for k in radd:
        n = nrb[k]
        out.append(f"| {n['table']} {k[0]} | {k[1]} | 새 행 | {'; '.join(f'{c}: {v}' for c, v in n['cells'].items())} |")
    for k in rdel:
        out.append(f"| {orb[k]['table']} {k[0]} | {k[1]} | 없어짐 | |")

    # 같은 질문, 두 문서
    if a.questions:
        from rag.search import Searcher
        so, sn = Searcher(a.old, cfg), Searcher(a.new, cfg)
        out += ["", "## 같은 질문의 판정 차이\n", f"| 문항 | {a.old} 판본 · 상태 | {a.new} 판본 · 상태 | 상위 쪽 {a.old} | 상위 쪽 {a.new} |",
                "|---|---|---|---|---|"]
        n_diff = 0
        for q in read_jsonl(Path(a.questions)):
            ro, rn = so.retrieve(q["question"]), sn.retrieve(q["question"])
            vo, vn = ro["version"], rn["version"]
            po, pn = [h["page"] for h in ro["hits"][:5]], [h["page"] for h in rn["hits"][:5]]
            if (vo["applied_version"], vo["status"]) != (vn["applied_version"], vn["status"]) or po != pn:
                n_diff += 1
                out.append(f"| {q['id']} | {vo['applied_version'] or '–'} · {vo['status']} | {vn['applied_version'] or '–'} · {vn['status']}"
                           f" | {po} | {pn} |")
        out.append(f"\n달라진 문항 {n_diff}개")

    text = "\n".join(out)
    path = ROOT / "results" / f"diff_{a.old}_{a.new}.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(text[:3000] + ("\n…(생략)" if len(text) > 3000 else ""))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
