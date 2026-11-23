# A의 판본 규칙(rag.version.decide)으로 문항마다 적용 판본을 정해 파일로 저장한다. 노트북의 VERSION_FILE로 쓴다.
# 실행: cd /workspace/dlthon2-rag && source .venv/bin/activate && python /workspace/DLthon02/ver3/make_version.py 질문.jsonl 판본.jsonl [정답.csv]
# 정답 CSV를 주면 적용_판본 정답과 비교한 점수와 틀린 문항을 같이 출력한다(개발 세트에만 쓴다. 봉인·공개 세트에는 주지 않는다).
import csv
import json
import os
import sys

sys.path.insert(0, os.getcwd())  # A의 저장소 폴더에서 실행해야 rag를 찾는다
from rag.common import ROOT
from rag.query import clean_query
from rag.version import decide

TO_TEAM = {"추가 확인": "미정", "규정 범위 밖": "해당 없음"}  # 팀 표기. 규정 범위 밖 = 사건일이 최초 시행일 전(예시 3과 같은 처리)

questions, out_path = sys.argv[1], sys.argv[2]
gold = {}
if len(sys.argv) > 3:
    gold = {r["문항번호"]: r["적용_판본"] for r in csv.DictReader(open(sys.argv[3], encoding="utf-8-sig"))}

n, ok, wrong = 0, 0, []
with open(questions, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
    for line in f:
        if not line.strip():
            continue
        q = json.loads(line)
        r = decide(clean_query(q["질문"], ROOT / "data" / "V2"), None, "V2")
        label = r.get("label") or r.get("applied_version") or r.get("status")
        team = TO_TEAM.get(label, label)
        out.write(json.dumps({"문항번호": q["문항번호"], "판본": team, "규칙_표기": label, "항목": r.get("item"),
                              "기준일": r.get("basis_date"), "이유": r.get("reason")}, ensure_ascii=False, default=str) + "\n")
        n += 1
        if q["문항번호"] in gold:
            ok += team == gold[q["문항번호"]]
            if team != gold[q["문항번호"]]:
                wrong.append(f"{q['문항번호']}(정답 {gold[q['문항번호']]}, 규칙 {team})")
print(f"저장: {out_path} ({n}문항)")
if gold:
    print(f"적용_판본 정답 {ok}/{n}")
    print("틀린 문항:", ", ".join(wrong) or "없음")
