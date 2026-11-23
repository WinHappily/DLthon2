# A의 검색(rag.search)으로 개발문항을 돌려, 노트북이 읽는 형식의 검색 결과 파일을 만든다.
# 실행: cd /workspace/dlthon2-rag && source .venv/bin/activate && python /workspace/DLthon02/ver3/make_t2.py 질문.jsonl 결과.jsonl
import json
import os
import re
import sys

sys.path.insert(0, os.getcwd())  # A의 저장소 폴더에서 실행해야 rag를 찾는다
from rag.brief import brief_text, build_brief
from rag.search import Searcher

QUESTIONS = sys.argv[1] if len(sys.argv) > 2 else "/workspace/DLthon02/ver2/data/개발문항_50.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/DLthon02/ver2/data/검색결과_T2_50.jsonl"
K = 5  # A의 README 기준값


def two_digit(name):
    """'표 7' → '표 07' (프롬프트의 표기 규칙과 맞춘다)"""
    return re.sub(r"표\s*(\d+)", lambda m: f"표 {int(m.group(1)):02d}", name)


searcher = Searcher("V2")
n = 0
with open(QUESTIONS, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as out:
    for line in f:
        if not line.strip():
            continue
        q = json.loads(line)
        res = searcher.retrieve(q["질문"], k=K)
        chunks = []
        for h in res["hits"]:
            names = [str(x) for x in (h.get("articles") or []) + (h.get("tables") or [])]
            why = h.get("why")
            chunks.append({
                "출처": two_digit("·".join(names)) or f"{h['page']}쪽",
                "쪽": h["page"],
                "본문": h["text"],
                "경로": "·".join(map(str, why)) if isinstance(why, list) else (why or "검색"),
            })
        # 판본_예측(team_label이 팀 표기)과 메모는 지금 노트북이 읽지 않는다. 메모 켬/끔 실험은 같은 파일로 노트북 설정만 바꾼다.
        row = {"문항번호": q["문항번호"], "조각": chunks, "판본_예측": res.get("version"),
               "메모": brief_text(build_brief(searcher, res))}
        out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        n += 1
print(f"저장: {OUT} ({n}문항)")
