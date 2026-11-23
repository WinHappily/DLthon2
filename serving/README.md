# serving — 모델 실행 · 비교 · 채점 (B · 지승환)

| 파일 | 역할 |
| --- | --- |
| `run_all.py` | 모델마다 vLLM 서버를 띄워 질문 파일을 일괄 실행하고 끈다. 결과는 한 줄에 한 문항 JSONL(검색 조각, 출력 원문, 시간, 토큰) |
| `평가_실행.sh` | 받기 → 실행 → 채점 → 지우기. 조건 `v2p3` · `v3p3` · `모두` · `점검` |
| `p3_V2.txt`, `p3_V3.txt`, `예시_p3.json` | 지시문 p3(상태 5가지 정의 + 판정 절차)와 예시 |
| `options_v2.json`, `options_v3.json` | 출력 스키마의 선택지(상태, 판본, 개정 항목, 기준일 종류) |
| `채점.py`, `score_cell.py` | 빠른 요약 채점, 유형별 표 · 혼동표 |
| `convert_a.py`, `make_questions.py`, `make_t2.py`, `make_oracle_page.py`, `make_version.py` | A의 검색 결과를 실행기 입력으로 변환, 오라클 조건 만들기 |
| `teach.sh`, `make_sft.py`, `train_lora.py`, `merge_lora.py` | 교사 답 생성 → 학습 데이터 → LoRA 학습 → 병합 |
| `prune_layers.py` | 층 제거 초안(검증 전 중단, 근거는 `../docs/설계.md`) |
| `to_submission.py` | 제출 CSV로 펼치기(R3 적용 포함) |
| `모델비교_시각화.py` | 19개 모델 비교 그림 |
| `읽어주세요_인수인계_B.md`, `실행표_1119.md` | 실행 방법 전체와 11/19 실행 기록 |

`data/`에는 개발 150 · 잡음 90 · 형식 30 질문과 정답, A 판본 규칙의 판본 파일이 있다. 검색 결과 파일은 용량 때문에 넣지 않았다(V3용은 `../data/retrieval_V3/`).

`train_lora.py`는 최대 길이 16,384토큰, `warmup_steps`로 고친 최종판이다(8,192에서는 746문항 중 9문항만 남았고, transformers 5.22에서 `warmup_ratio`가 오류).
