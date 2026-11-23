# 교사 답(run_all 결과)을 증류(LoRA SFT)용 대화 파일로 바꾼다. 학생이 실제로 받을 입력과 똑같이 run_all 함수로 만든다
# 사용: python make_sft.py <교사 결과.jsonl> <교사 모델ID> <질문.jsonl> <검색결과.jsonl> <판본.jsonl> <지시문.txt> <출력.jsonl> [남길 문항번호 목록.txt]
#   예: python make_sft.py results/results_TEACHp3_860_V3.jsonl gemma4-31b-qat data/문항_train860_V3.jsonl \
#         data/검색결과_train860_V3.jsonl data/판본_train860_V3.jsonl p3_V3.txt data/sft_V3.jsonl C가_고른_문항.txt
#   끝까지 쓴(stop) 답 중 JSON이 맞는 것만 쓴다. 목록 파일을 주면 그 문항만(한 줄에 문항번호 하나)
import json, sys
import run_all as R
res, model, qf, rf, vf, sysf, out = sys.argv[1:8]
keep = set(open(sys.argv[8], encoding="utf-8").read().split()) if len(sys.argv) > 8 else None
ans = {}
for line in open(res, encoding="utf-8"):
    r = json.loads(line)
    if r["모델ID"] == model and r["오류"] is None and r["종료_사유"] == "stop":
        try:
            json.loads(r["출력_원문"]); ans[r["문항번호"]] = r["출력_원문"]
        except Exception:
            pass
qs = {q["문항번호"]: q for q in R.load_jsonl(qf)}
ret = R.load_retrieval(rf, None)
memo = {r["문항번호"]: r.get("메모") for r in R.load_jsonl(rf)}
ver = {r["문항번호"]: r for r in R.load_jsonl(vf)}
system = open(sysf, encoding="utf-8").read().format(examples=R.format_examples("예시_p3.json"))
n = 0
with open(out, "w", encoding="utf-8") as f:
    for q in qs:
        if q not in ans or (keep is not None and q not in keep):
            continue
        user = R.format_user(qs[q]["질문"], ret.get(q, []), memo.get(q), ver.get(q))
        f.write(json.dumps({"문항번호": q, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                                                     {"role": "assistant", "content": ans[q]}]}, ensure_ascii=False) + "\n")
        n += 1
print(f"{out}: {n}문항")
