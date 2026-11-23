# 판본 규칙 효과 검증 (분석용, 최종 시스템은 바꾸지 않음)
# 봉인 50에서 '판본 규칙 없이' 돌린다: 판정 메모의 판본 줄(- 판본 : …, - 판본 대조 : …)을 지우고 판본 고정도 끈다.
# 검색 조각과 나머지 메모(금액 등)는 그대로라, 차이는 '규칙이 정한 판본을 주느냐'만 남는다
cd "$(dirname "$0")"
python - <<'P' || exit 1
import json
for V in ("V2", "V3"):
    n = 0
    with open(f"data/검색결과_SEAL_{V}_무규칙.jsonl", "w", encoding="utf-8") as f:
        for l in open(f"data/검색결과_SEAL_{V}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            lines = (r.get("메모") or "").split("\n")
            kept = [x for x in lines if not x.startswith("- 판본")]
            n += len(lines) - len(kept)
            r["메모"] = "\n".join(kept)
            r.pop("판본_예측", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{V}: 판본 줄 {n}개 지움")
P
COMMON="--questions data/봉인문항_50.jsonl --qset sealed50 --models gemma4-31b-qat,gemma4-12b-qat --startup-timeout 900 --request-timeout 600 --concurrency 8 --use-memo --examples 예시_p3.json --max-tokens 2048"
python run_all.py $COMMON --retrieval data/검색결과_SEAL_V2_무규칙.jsonl --retrieval-ver a-g3.3-bge-no-version --exp SEALnoR_V2 --out results/results_SEALnoR_V2.jsonl --system-file p3_V2.txt
python run_all.py $COMMON --retrieval data/검색결과_SEAL_V3_무규칙.jsonl --retrieval-ver a-g3.3-bge-no-version --exp SEALnoR_V3 --out results/results_SEALnoR_V3.jsonl --system-file p3_V3.txt --options options_v3.json
echo "무규칙 실행 끝: results/results_SEALnoR_V2.jsonl, results/results_SEALnoR_V3.jsonl"
