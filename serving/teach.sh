# 교사 답 만들기: A의 학습 질문 검색 결과(V2판·V3판)를 받아 교사 모델로 답을 만든다. 출력은 원본 JSON 전체(C가 거름)
# 사용: bash teach.sh <A의 retrieval_train860_V2_g32.jsonl> <A의 retrieval_train860_V3_g32.jsonl> [모델ID 저장소]
#   기본 교사: gemma4-31b-qat google/gemma-4-31B-it-qat-w4a16-ct
# 결과: results/results_TEACHp3_860_V2.jsonl, results/results_TEACHp3_860_V3.jsonl  (끊겨도 다시 실행하면 이어서)
cd "$(dirname "$0")"
F2=$1; F3=$2; MID=${3:-gemma4-31b-qat}; REPO=${4:-google/gemma-4-31B-it-qat-w4a16-ct}
M_DIR=/workspace/models
for d in V2 V3; do
  f=$F2; [ $d = V3 ] && f=$F3
  python make_questions.py $f data/문항_train860_$d.jsonl || exit 1
  python convert_a.py $f data/검색결과_train860_$d.jsonl data/판본_train860_$d.jsonl data/문항_train860_$d.jsonl || exit 1
done
python -c "from huggingface_hub import snapshot_download as s; s('$REPO', local_dir='$M_DIR/$MID')" || { echo "모델 받기 실패"; exit 1; }
for d in V2 V3; do
  O="--examples 예시_p3.json --system-file p3_$d.txt"; [ $d = V3 ] && O="$O --options options_v3.json"
  python run_all.py --questions data/문항_train860_$d.jsonl --retrieval data/검색결과_train860_$d.jsonl --retrieval-ver a-g32 \
    --exp TEACHp3_860_$d --qset train860 --out results/results_TEACHp3_860_$d.jsonl --models $MID \
    --startup-timeout 900 --request-timeout 600 --concurrency 8 --use-memo --version-file data/판본_train860_$d.jsonl $O
done
[ "$KEEP" = 1 ] || rm -rf "$M_DIR/$MID"
python - <<'PY'
import json
for d in ("V2", "V3"):
    rows = [json.loads(l) for l in open(f"results/results_TEACHp3_860_{d}.jsonl", encoding="utf-8")]
    ok = sum(r["오류"] is None and r["종료_사유"] == "stop" for r in rows)
    print(f"TEACHp3_860_{d}: {len(rows)}행, 끝까지 쓴 답 {ok}")
PY
