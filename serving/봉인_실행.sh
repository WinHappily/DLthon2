# 봉인 50 실행기 (딱 한 번만): A 검색 파일에서 질문 파일 만들기 → 변환 → 두 모델 × V2·V3 실행
# 준비: data/retrieval_sealed50_V2.jsonl, data/retrieval_sealed50_V3.jsonl (A가 준 파일)
# 사용: bash 봉인_실행.sh   (끊겨도 다시 실행하면 끝난 문항은 건너뜀). 채점은 봉인 채팅에서 한다
cd "$(dirname "$0")"
mkdir -p results
# 1) 질문 파일: A 검색 파일 안의 질문 원문을 그대로 꺼낸다 (화면에 출력하지 않음)
python -c "
import json
with open('data/봉인문항_50.jsonl', 'w', encoding='utf-8') as f:
    for l in open('data/retrieval_sealed50_V2.jsonl', encoding='utf-8'):
        r = json.loads(l); f.write(json.dumps({'문항번호': r['id'], '질문': r['question']}, ensure_ascii=False) + '\n')
" || exit 1
# 2) A 검색 결과 → 우리 형식. V3 파일의 질문이 V2 파일과 같은지도 확인
python convert_a.py data/retrieval_sealed50_V2.jsonl data/검색결과_SEAL_V2.jsonl data/판본_A규칙_SEAL_V2.jsonl data/봉인문항_50.jsonl || exit 1
python convert_a.py data/retrieval_sealed50_V3.jsonl data/검색결과_SEAL_V3.jsonl data/판본_A규칙_SEAL_V3.jsonl data/봉인문항_50.jsonl || exit 1
[ -d /workspace/models/gemma4-12b-qat ] || python -c "from huggingface_hub import snapshot_download as s; s('google/gemma-4-12B-it-qat-w4a16-ct', local_dir='/workspace/models/gemma4-12b-qat')" || exit 1
COMMON="--questions data/봉인문항_50.jsonl --qset sealed50 --models gemma4-12b-qat,qwen3-4b-lora --startup-timeout 900 --request-timeout 600 --concurrency 8 --use-memo --examples 예시_p3.json --max-tokens 2048"
python run_all.py $COMMON --retrieval data/검색결과_SEAL_V2.jsonl --retrieval-ver a-g3.3-bge --exp SEAL_V2 --out results/results_SEAL_V2.jsonl \
  --version-file data/판본_A규칙_SEAL_V2.jsonl --system-file p3_V2.txt
python run_all.py $COMMON --retrieval data/검색결과_SEAL_V3.jsonl --retrieval-ver a-g3.3-bge --exp SEAL_V3 --out results/results_SEAL_V3.jsonl \
  --version-file data/판본_A규칙_SEAL_V3.jsonl --system-file p3_V3.txt --options options_v3.json
echo "봉인 실행 끝: results/results_SEAL_V2.jsonl, results/results_SEAL_V3.jsonl"
