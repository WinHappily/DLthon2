# 허RAG해줘 · DLthon2 — 모델은 줄이고, 판단은 지킨다

신혼생활 공동생활 규약집(V2 · V3)을 근거로 질문에 답하는 RAG 상담 서비스에서, **모델을 줄여도 판정 정확도를 지킬 수 있는지** 실험한 저장소입니다.

- 팀: 권해나(A · 문서·검색·판본 규칙) · 지승환(B · 모델·서빙·실험) · 이승화(C · 팀장, 평가·기록·발표)
- 환경: RunPod A40 48GB 1장 · vLLM 0.34.0
- 실험 기록 전체: [docs/실험기록.md](docs/실험기록.md)

## 결과 한눈에 (개발 150, 판정 상태와 적용 판본이 모두 맞은 문항 수)

| 구성 | V2 / V3 | 가중치 메모리 | 응답 중앙값 / p95 | 역할 |
| --- | --- | --- | --- | --- |
| 기준선 qwen3-32b 4비트(AWQ) + R3 | 93 / 95 | 18.2 GiB | 67초 / 95초 | 기준 |
| **gemma4-12b 4비트(QAT) + R3** | **100 / 98** | **8.3 GiB** | **26초 / 39초** | 주력 구성 |
| **qwen3-4b + LoRA 증류(교사 31B)** | **95 / 95** (증류 전 72 / 63) | 7.6 GiB | 16초 / 24초 | 극한 경량화 |
| gemma4-31b 4비트(QAT) | 121 / 113 | 19.8 GiB | 58초 / 94초 | 참고 · 자원이 충분할 때 |

응답 시간은 동시 8요청 중 측정(대기 포함). 처음 보는 질문(잡음 90 · 형식 30 · 공개 100 · 봉인 50) 결과와 한계는 [docs/실험기록.md](docs/실험기록.md)에 있습니다.

## 설계

```
질문 → [코드] 검색(bge-m3) + 판정 메모 → [코드] 판본 결정(개정 항목 + 기준일 → V1·V2·V3)
     → [모델] 판정 JSON(상태 · 근거 · 답 · 다음 행동) → [코드] 후처리 R3
```

- 답이 하나로 정해지는데 틀리면 판정이 뒤집히는 일(판본 결정, 후처리)은 코드가, 조항 해석과 판정은 모델이 맡는다.
- R3: 상태가 '추가 확인'이고 빠진 정보가 합의 여부뿐이며 아직 하지 않은 일이면 → '사전 합의 필요'. 기준선을 포함한 모든 모델에 똑같이 적용해 비교했다.

## 폴더

| 폴더 | 내용 | 담당 |
| --- | --- | --- |
| `retrieval/` | PDF 추출(본문·표, 쪽 번호 보존), 쪽 단위 청크 + 꼬리표, 표 행 보조 청크, bge-m3 색인·검색, 판본 규칙(`rag/version.py`), 검색 평가 | A |
| `serving/` | vLLM 일괄 실행기 `run_all.py`, 지시문 `p3_V2.txt`·`p3_V3.txt`, 출력 스키마 `options_v*.json`, 채점 `채점.py`, 개발 150 · 잡음 90 · 형식 30 문항과 정답 | B |
| `distill/` | 증류 학습 질문 생성 스크립트, 평가 세트와의 겹침 검사(`overlap_check.py`와 결과), 채점을 통과한 교사 답 목록(`C가_고른_문항_V*.txt`). 실행 스크립트(`teach.sh`, `make_sft.py`, `train_lora.py`, `merge_lora.py`)는 실행기와 같은 폴더인 `serving/`에 있음 | C |
| `eval/` | 공개 100 · 봉인 50 실행과 채점, 판본 규칙을 끈 분석 실행 | C |
| `results/` | 요약 채점표(개발 · 잡음 · 형식 · 교사 답 · 공개 100) | C |

## 재현 방법

```bash
# 0) 모델 받기 (gemma는 Hugging Face에서 라이선스 동의 필요, 토큰은 터미널에서만 export)
python -c "from huggingface_hub import snapshot_download as s; s('google/gemma-4-12B-it-qat-w4a16-ct', local_dir='/workspace/models/gemma4-12b-qat')"

# 1) 평가 (serving/) — V2·V3 개발 150 · 잡음 90 · 형식 30
cd serving
bash 평가_실행.sh gemma4-12b-qat google/gemma-4-12B-it-qat-w4a16-ct 모두
python 채점.py            # → results/평가_요약.md

# 2) 증류 (serving/에서 실행) — 교사 답 → 거르기(distill/C가_고른_문항_V3.txt) → 학습 → 병합 → 평가
bash teach.sh retrieval_train860_V2_g33.jsonl retrieval_train860_V3_g33.jsonl
python make_sft.py ...    # 인자는 serving/읽어주세요_인수인계_B.md 'LoRA 학습' 참고
python train_lora.py data/sft_V3.jsonl /workspace/models/qwen3-4b-instruct-2507 /workspace/lora_qwen3-4b
python merge_lora.py /workspace/models/qwen3-4b-instruct-2507 /workspace/lora_qwen3-4b /workspace/models/qwen3-4b-lora
bash 평가_실행.sh qwen3-4b-lora - 모두
```

- 검색 결과 파일(`검색결과_*.jsonl`, `retrieval_*.jsonl`)은 `retrieval/`의 노트북(`notebooks/role_A_colab.ipynb`)으로 만들어 `serving/data/`에 넣는다.
- vLLM은 직접 띄우지 않는다. `run_all.py`가 모델마다 서버를 띄우고 끈다. 두 개를 동시에 띄우지 않는다.

## 모델 · 체크포인트 식별자

| 용도 | 식별자 |
| --- | --- |
| 기준선 | `Qwen/Qwen3-32B-AWQ` |
| 주력 | `google/gemma-4-12B-it-qat-w4a16-ct` + 후처리 R3 |
| 교사 / 참고 | `google/gemma-4-31B-it-qat-w4a16-ct` |
| 학생 | `Qwen/Qwen3-4B-Instruct-2507` + LoRA(랭크 16 · alpha 32 · dropout 0.05 · 모든 선형층 · 학습률 1e-4 · 2에폭 · 최대 16,384토큰, 교사 답 746문항) |

생성 설정: temperature 0 · seed 0 · 추론 모드 끔 · JSON 스키마 강제 · 최대 출력 1,536토큰(형식·공개·봉인 2,048) · 최대 문맥 32,768 · GPU 메모리 90% · 동시 8요청.

## 규약집 인덱싱 방법

PDF에서 본문과 표를 따로 추출하고 쪽 번호를 보존 → 쪽 단위 청크에 꼬리표(판본·쪽·조항·표·개정 항목) → 표의 각 행은 열 이름을 붙인 보조 청크로 분리 → bge-m3 임베딩 색인, 재순위 후 청크 안의 '제N조 · 표 NN'을 한 단계 따라가 함께 가져옴. V2 합본과 V3 합본은 색인을 따로 만들고, 개정 비교 때 색인만 바꿔 끼운다.

## 넣지 않은 것

- 규약집 원문과 그로부터 만든 청크 · 검색 결과 파일(운영진 배포 자료)
- 운영진 공개 100 · 평가 키트의 질문과 정답, 팀장이 보관한 봉인 50 문항과 정답
- 대용량 모델 출력 원본(JSONL)과 LoRA 체크포인트
