# serving — 모델 실행 · 비교 · 채점 (B · 지승환)

| 파일 | 역할 |
| --- | --- |
| `run_all.py` | 모델마다 vLLM 서버를 띄워 질문 파일을 일괄 실행하고 끈다. 결과는 한 줄에 한 문항 JSONL(검색 조각, 출력 원문, 시간, 토큰) |
| `평가_실행.sh` | 받기 → 실행 → 채점 → 지우기. 조건 `v2p3` · `v3p3` · `모두` · `점검` |
| `p3_V2.txt`, `p3_V3.txt`, `예시_p3.json` | 지시문 p3(상태 3가지 정의(사전 합의 필요 · 추가 확인 · 공동 처리 불가) + 판정 절차 + 예시 2개) |
| `options_v2.json`, `options_v3.json` | 출력 스키마의 선택지(상태, 판본, 개정 항목, 기준일 종류) |
| `채점.py`, `score_cell.py` | 빠른 요약 채점, 유형별 표 · 혼동표 |
| `convert_a.py`, `make_questions.py`, `make_t2.py`, `make_oracle_page.py`, `make_version.py` | A의 검색 결과를 실행기 입력으로 변환, 오라클 조건 만들기 |
| `teach.sh`, `make_sft.py`, `train_lora.py`, `merge_lora.py` | 교사 답 생성 → 학습 데이터 → LoRA 학습 → 병합 |
| `prune_layers.py` | 층 제거 초안(검증 전 중단, 근거는 `../docs/설계.md`) |
| `to_submission.py` | 제출 CSV로 펼치기(R3 적용 포함) |
| `모델비교_시각화.py` | 19개 모델 비교 그림 |
| `공개_실행.sh`, `채점_PUB.py`, `봉인_실행.sh`, `채점_봉인.py`, `무규칙_봉인.sh` | 공개 100 · 봉인 50 실행과 채점, 판본 규칙을 끈 분석 실행(C). 질문 · 정답 파일은 저장소에 없음 |
| `읽어주세요_인수인계_B.md`, `실행표_1119.md` | 실행 방법 전체와 11/19 실행 기록 |

`data/`에 평가에 필요한 입력이 모두 있다.

| 파일 | 내용 |
| --- | --- |
| `개발문항_150.jsonl`, `잡음문항_90.jsonl`, `형식문항_30.jsonl`, `점검문항_5.jsonl` | 질문 |
| `개발정답_150_N90_F30.csv` | V2 정답(V3 정답은 `../data/eval_sets/V3/`) |
| `검색결과_T2_*.jsonl`, `판본_A규칙_*.jsonl` | V2 문서 검색 결과(판정 메모 포함)와 판본 규칙 g2.2 판정 |
| `검색결과_T2v3_*.jsonl`, `판본_A규칙v3_*.jsonl` | V3 문서 검색 결과와 판본 규칙 g3.3으로 다시 만든 판정(실험 당시 g3.0 판정과 150문항 모두 같음, A 검색 원본 `../data/retrieval_V3/`를 `convert_a.py`로 변환) |
| `검색결과_oracle_*.jsonl` | 오라클 조건(정답 근거 쪽을 직접 넣음) |

`평가_실행.sh`는 `data/검색결과_T2v3_150.jsonl`이 있으면 V3 실험을 g3.3 검색으로 돌린다(실험ID `T2MRp3_V3_*`, 결과 `../results/model_outputs/`와 같은 조건).

`train_lora.py`는 최대 길이 16,384토큰, `warmup_steps`로 고친 최종판이다(8,192에서는 746문항 중 9문항만 남았고, transformers 5.22에서 `warmup_ratio`가 오류).
