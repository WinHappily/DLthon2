# A의 검색 결과 파일(retrieval_*.jsonl)에서 run_all이 읽는 질문 파일(문항번호, 질문)을 만든다
# 사용: python make_questions.py <A 파일> <질문 파일 출력>
import json, sys
src, out = sys.argv[1:3]
n = 0
with open(out, "w", encoding="utf-8") as f:
    for line in open(src, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            f.write(json.dumps({"문항번호": r["id"], "질문": r["question"]}, ensure_ascii=False) + "\n")
            n += 1
print(f"{out}: {n}문항")
