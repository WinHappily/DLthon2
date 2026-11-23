# 공개 100 실행기: A 검색 변환 → (12b 없으면 받기) → 두 모델 × V2·V3 실행 → 채점_PUB.py
# 사용: bash 공개_실행.sh      (끊겨도 다시 실행하면 끝난 문항은 건너뜀)
cd "$(dirname "$0")"
mkdir -p results
# 1) A 검색 결과 → 우리 형식(검색 조각 + 메모, 판본 고정 파일). 질문 원문이 같은지도 확인
python convert_a.py data/retrieval_public100_V2.jsonl data/검색결과_PUB_V2.jsonl data/판본_A규칙_PUB_V2.jsonl data/공개문항_100.jsonl || exit 1
python convert_a.py data/retrieval_public100_V3.jsonl data/검색결과_PUB_V3.jsonl data/판본_A규칙_PUB_V3.jsonl data/공개문항_100.jsonl || exit 1
# 2) 최종 모델 gemma4-12b가 지워져 있으면 다시 받기 (증류 모델은 /workspace/models/qwen3-4b-lora 에 있음)
[ -d /workspace/models/gemma4-12b-qat ] || python -c "from huggingface_hub import snapshot_download as s; s('google/gemma-4-12B-it-qat-w4a16-ct', local_dir='/workspace/models/gemma4-12b-qat')" || exit 1
COMMON="--questions data/공개문항_100.jsonl --qset pub100 --models gemma4-12b-qat,qwen3-4b-lora --startup-timeout 900 --request-timeout 600 --concurrency 8 --use-memo --examples 예시_p3.json --max-tokens 2048"
# 3) V2 문서
python run_all.py $COMMON --retrieval data/검색결과_PUB_V2.jsonl --retrieval-ver a-g3.3-bge --exp PUB_V2 --out results/results_PUB_V2.jsonl \
  --version-file data/판본_A규칙_PUB_V2.jsonl --system-file p3_V2.txt
# 4) V3 문서
python run_all.py $COMMON --retrieval data/검색결과_PUB_V3.jsonl --retrieval-ver a-g3.3-bge --exp PUB_V3 --out results/results_PUB_V3.jsonl \
  --version-file data/판본_A규칙_PUB_V3.jsonl --system-file p3_V3.txt --options options_v3.json
# 5) 채점
python 채점_PUB.py
