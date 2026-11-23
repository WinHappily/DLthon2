"""A가 보낸 검색 결과(retrieval_<이름>_V2.jsonl, 판정 메모 있는 판)를 노트북이 읽는 두 파일로 바꾼다.
  1) T2 검색 결과: {문항번호, 조각[출처·쪽·본문·경로], 메모, 판본_예측}  → RETRIEVAL (메모 켬/끔은 USE_MEMO로)
  2) 판본 파일:   {문항번호, 판본, 규칙_표기, 항목, 기준일, 이유}       → VERSION_FILE
A 파일의 messages는 쓰지 않는다(A 지시문이라 우리 형식과 다름).
실행: python convert_a.py <A 검색결과.jsonl> <T2 출력.jsonl> <판본 출력.jsonl> [질문.jsonl]
질문 파일을 주면 A가 쓴 질문 원문이 우리 질문 파일과 같은지 확인한다(다르면 검색을 다시 받아야 함).
"""
import json
import re
import sys

TO_TEAM = {"추가 확인": "미정", "규정 범위 밖": "해당 없음"}


def two_digit(name):  # '표 7' → '표 07' (make_t2와 같은 표기)
    return re.sub(r"표\s*(\d+)", lambda m: f"표 {int(m.group(1)):02d}", name)


src, out_t2, out_ver = sys.argv[1:4]
rows = [json.loads(line) for line in open(src, encoding="utf-8") if line.strip()]
if len(sys.argv) > 4:
    q = {json.loads(l)["문항번호"]: json.loads(l)["질문"] for l in open(sys.argv[4], encoding="utf-8") if l.strip()}
    diff = [r["id"] for r in rows if q.get(r["id"]) != r["question"]]
    missing = sorted(set(q) - {r["id"] for r in rows})
    print(f"질문 대조: 다른 문항 {len(diff)}개 {diff[:15]}, 검색 결과에 없는 문항 {len(missing)}개 {missing[:5]}")
with open(out_t2, "w", encoding="utf-8") as f2, open(out_ver, "w", encoding="utf-8") as fv:
    for r in rows:
        chunks = [{"출처": two_digit("·".join([str(x) for x in (h.get("articles") or [])] + [str(x) for x in (h.get("tables") or [])]))
                           or f"{h['page']}쪽",
                   "쪽": h["page"], "본문": h["text"],
                   "경로": "·".join(map(str, h["why"])) if isinstance(h.get("why"), list) else (h.get("why") or "검색")}
                  for h in r["hits"]]
        v = r["version"]
        raw = v.get("label") or v.get("applied_version") or v.get("status")
        team = TO_TEAM.get(v.get("team_label") or raw, v.get("team_label") or raw)
        f2.write(json.dumps({"문항번호": r["id"], "조각": chunks, "메모": r.get("brief_text"), "판본_예측": v},
                            ensure_ascii=False, default=str) + "\n")
        fv.write(json.dumps({"문항번호": r["id"], "판본": team, "규칙_표기": v.get("label_raw") or raw, "항목": v.get("item"),
                             "기준일": v.get("basis_date"), "이유": v.get("reason")}, ensure_ascii=False, default=str) + "\n")
print(f"저장: {out_t2}, {out_ver} ({len(rows)}문항, 메모 {'있음' if rows and rows[0].get('brief_text') else '없음 — 메모 있는 판을 넣으세요'})")
