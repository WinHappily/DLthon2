# 평가 실행기: 모델 받기 → A 검색 + 판정 메모 + 판본 고정(T2MR) 조건으로 돌리기 → 채점 → 모델 지우기
# 사용: bash 평가_실행.sh <모델ID> <HF 저장소 또는 -> <조건>
#   조건: v2p3 = T2MRp3_150 · T2MRp3_N90        (V2 문서, p3 지시문, 예시 2개)
#         v3p3 = T2MRp3_V3_150 · _N90 · _F30     (V3 문서, p3_V3)
#         v2p2 = T2MR_150 · T2MR_N90             (11/19 기존 조건 그대로)
#         점검 = CHECK5 (5문항, p2. 환경 확인용)
#         모두 = v2p3 + v3p3
#   저장소 자리에 - 를 주면 받지 않고 /workspace/models/<모델ID>를 그대로 쓴다 (증류한 모델 등)
#   모델을 지우지 않으려면 앞에 KEEP=1
# 결과: results/results_<실험ID>.jsonl, 요약: results/평가_요약.md. 끊겨도 다시 실행하면 끝난 문항은 건너뜀
cd "$(dirname "$0")"
MID=$1; REPO=$2; C=${3:-v3p3}
M_DIR=/workspace/models   # run_all.py가 모델을 찾는 곳
if ! python -c "import run_all, sys; sys.exit(0 if '$MID' in run_all.MODELS else 1)"; then
  echo "등록된 모델ID가 아닙니다. 쓸 수 있는 ID:"; python -c "import run_all; print(' '.join(run_all.MODELS))"; exit 1
fi
case $C in v2p3|v3p3|v2p2|점검|모두) ;; *) echo "조건은 v2p3 · v3p3 · v2p2 · 점검 · 모두 중 하나"; exit 1 ;; esac
mkdir -p $M_DIR results
if [ "$REPO" != - ]; then
  python -c "from huggingface_hub import snapshot_download as s; s('$REPO', local_dir='$M_DIR/$MID')" || { echo "모델 받기 실패 (gemma는 HF 로그인·라이선스 동의 필요)"; exit 1; }
fi
run() {  # 실험ID 질문 검색결과 검색버전 질문세트 [추가 인자]
  python run_all.py --questions data/$2.jsonl --retrieval data/$3.jsonl --retrieval-ver $4 \
    --exp $1 --qset $5 --out results/results_$1.jsonl --models $MID --startup-timeout 900 --request-timeout 600 \
    --concurrency 8 --use-memo "${@:6}"
}
P2="--examples 예시_p2.json"
P3V2="--examples 예시_p3.json --system-file p3_V2.txt"
P3V3="--examples 예시_p3.json --system-file p3_V3.txt --options options_v3.json"
if [ -s data/검색결과_T2v3_150.jsonl ]; then V3=V3; R3=T2v3; RV3=a-g3-bge; else V3=V3bm; R3=T2v3bm; RV3=a-g3.0-bm25; fi
if [ $C = v2p3 ] || [ $C = 모두 ]; then
  run T2MRp3_150 개발문항_150 검색결과_T2_150 a-g2.2-bge dev150v2 --version-file data/판본_A규칙_150.jsonl $P3V2
  run T2MRp3_N90 잡음문항_90 검색결과_T2_잡음_90 a-g2.2-bge noise90 --version-file data/판본_A규칙_잡음_90.jsonl $P3V2
fi
if [ $C = v3p3 ] || [ $C = 모두 ]; then
  run T2MRp3_${V3}_150 개발문항_150 검색결과_${R3}_150 $RV3 dev150v2 --version-file data/판본_A규칙v3_150.jsonl $P3V3
  run T2MRp3_${V3}_N90 잡음문항_90 검색결과_${R3}_잡음_90 $RV3 noise90 --version-file data/판본_A규칙v3_잡음_90.jsonl $P3V3
  run T2MRp3_${V3}_F30 형식문항_30 검색결과_${R3}_형식_30 $RV3 f30 --version-file data/판본_A규칙v3_형식_30.jsonl $P3V3 --max-tokens 2048
fi
if [ $C = v2p2 ]; then
  run T2MR_150 개발문항_150 검색결과_T2_150 a-g2.2-bge dev150v2 --version-file data/판본_A규칙_150.jsonl $P2
  run T2MR_N90 잡음문항_90 검색결과_T2_잡음_90 a-g2.2-bge noise90 --version-file data/판본_A규칙_잡음_90.jsonl $P2
fi
if [ $C = 점검 ]; then
  run CHECK5 점검문항_5 검색결과_T2_150 a-g2.2-bge check5 --version-file data/판본_A규칙_150.jsonl $P2 --concurrency 1
fi
[ "$KEEP" = 1 ] || [ "$REPO" = - ] || rm -rf "$M_DIR/$MID"
python 채점.py | tail -n +4
echo "요약: results/평가_요약.md"
