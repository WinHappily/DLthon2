# 인수인계 B (11/20 금 15시)

> 수정사항 존재하면 반드시 해당 날짜와 한줄로 요약해서 덧붙일것

B는 월요일 전까지 연락이 안 됩니다. 이 문서와 `ver5_1120.zip` 하나로 이어서 할 수 있게 정리했습니다.
명령은 전부 **팟 터미널에 그대로 붙여 넣으면** 됩니다. 한 칸(코드 상자)씩 붙여 넣으세요.

## 0. 사용법 (워크스페이스에 올리고 붙여 넣기만)

1. Jupyter 파일 창에서 `ver5_1120.zip`을 `/workspace/DLthon02`에 올립니다
2. 터미널에 붙여 넣습니다

```bash
cd /workspace/DLthon02 && python -c "import zipfile; zipfile.ZipFile('ver5_1120.zip').extractall('.')" && cd ver5 && ls
```

3. 하고 싶은 일의 명령을 붙여 넣습니다

| 하고 싶은 일 | 명령 (ver5 폴더에서) |
|---|---|
| 환경 확인 (5문항, 약 5분) | `bash 평가_실행.sh qwen3-4b-instruct-2507 Qwen/Qwen3-4B-Instruct-2507 점검` |
| 기준선 p3 (V2·V3) | `bash 평가_실행.sh qwen3-32b-awq Qwen/Qwen3-32B-AWQ 모두` |
| 학생 p3 (V2·V3) | `bash 평가_실행.sh qwen3-4b-instruct-2507 Qwen/Qwen3-4B-Instruct-2507 모두` |
| 최종 후보 p3 (V2·V3) | `bash 평가_실행.sh gemma4-12b-qat google/gemma-4-12B-it-qat-w4a16-ct 모두` |
| 교사 답 860 (V2·V3) | `bash teach.sh retrieval_train860_V2_g32.jsonl retrieval_train860_V3_g32.jsonl` |
| 결과 요약 보기 | `python 채점.py` → `results/평가_요약.md` |

- 모델은 `/workspace/models`에 받았다가 끝나면 지웁니다. 결과는 `ver5/results/`에 쌓입니다
- 끊기면 같은 명령을 다시 붙여 넣으면 끝난 문항은 건너뛰고 이어서 합니다
- **p3** = 지시문 3판입니다. p2는 11/19까지 모든 실험에 쓴 지시문이고, p3는 오늘 팀이 합의한 수정판입니다(② 참고)

### 폴더 규칙: 새 세트마다 verN

- ver3 = 11/19 낮, ver4 = 11/19 밤, **ver5 = 이 묶음(p3 · V3 · 교사 답)**
- 다음 세트(예: V3 정답 반영, LoRA 평가)는 ver6으로 새로 만듭니다. 코드와 데이터만 복사하고 결과는 비운 채 시작합니다

```bash
cd /workspace/DLthon02 && mkdir ver6 && cp -r ver5/*.py ver5/*.sh ver5/*.txt ver5/*.json ver5/data ver6/ && mkdir ver6/results && cd ver6 && ls
```

- 이전 세트의 결과는 그 폴더에 그대로 두고 덮어쓰지 않습니다

---

## 역할 분담 (주말 ~ 월)

| 사람 | 맡는 일 |
|---|---|
| **C** | **증류**: 교사 답 860(V2 · V3) → 거르기 → 학생 LoRA 학습 → 평가 ('교사 답 생성' · '⑤ LoRA 학습') |
| **A** | 검색 파일 공급: V3 개발 세트 · 학습 질문 860 · 공개 100 (bge) ('V3로 돌리는 법' · '③ 제출') |
| **B** | **경량 모델을 최대한 끌어올리는 방법** 찾기 · 다듬기 (맨 아래 'B 트랙'). 복귀 후 결과에 맞춰 적용하고 발표에 정리 |

- 증류(C)는 **모델 자체**를 바꾸고, B의 방법은 모델은 그대로 두고 **모델이 할 일을 줄이거나 코드가 대신**합니다. 둘을 함께 쓸 수 있습니다 (예: 증류한 4B + R3)

---

## ① 접근 권한

### RunPod

- 팟은 이미 팀에 공유되어 있습니다 (B도 같은 팟을 씁니다)
- 사양: GPU **A40 48GB** 1장, 볼륨 **100GB** (`/workspace`), vLLM 0.34
- 남은 크레딧 · 결제 계정은 팟 주인에게 확인합니다
- 대회 예산은 팀당 RunPod 크레딧 20달러입니다
- **끄는 법**: RunPod 웹 → Pods → 이 팟 → **Stop**
  - `/workspace`(볼륨)는 남고, 그 밖의 디스크는 지워집니다
  - Stop 상태에서도 볼륨 요금은 조금 나갑니다
  - **Terminate는 절대 누르지 마세요** (볼륨까지 다 지워집니다)
- 팟 안에서 끄기: `runpodctl stop pod $RUNPOD_POD_ID`

### Hugging Face

- 라이선스 동의가 필요한 모델 (각자 계정으로 모델 페이지에서 동의)
  - google/gemma-4-31B-it-qat-w4a16-ct (교사)
  - google/gemma-4-12B-it-qat-w4a16-ct (현재 최종 후보)
  - google/gemma-4-E4B-it, google/gemma-4-E2B-it
  - RedHatAI/gemma-4-31B-it-FP8-dynamic (gemma 파생이라 동의가 필요할 수 있음)
- Qwen · EXAONE · kanana는 동의 없이 받힙니다
- **토큰 넣는 곳**: 모델을 받는 그 터미널에서 실행합니다. 토큰을 문서나 채팅에 쓰지 마세요

```bash
export HF_TOKEN=여기에_본인_토큰
python -c "from huggingface_hub import whoami; print('HF 로그인:', whoami()['name'])"
```

- `export`는 그 터미널 창에서만 유효합니다. 새 터미널을 열면 다시 넣습니다

### GitHub

- B는 깃허브에 올리지 않았습니다. **이 압축(ver5)이 B 코드의 최신본**입니다

---

## ② p3 지시문 확정본 (오늘 합의 반영)

| 파일 | 쓰는 곳 |
|---|---|
| `p3_V2.txt` | V2 문서로 돌릴 때 |
| `p3_V3.txt` | V3 문서로 돌릴 때 ('V3 합본', 판본 수치 표 53~57) |
| `예시_p3.json` | 예시 1(가능) · 2(추가 확인)만. 예시 3(규약과 무관)은 뺐습니다. C가 '규정에 없음' 예시를 넣을 때 이 파일에 추가 |

p2에서 바뀐 곳:

1. 상태 정의 3줄 추가
   - 사전 합의 필요: 아직 하지 않은 앞으로의 행동인데, 필요한 합의·요청·동의가 없다
   - 추가 확인: 이미 일어난 일인데 판정을 가르는 사실을 모른다
   - 공동 처리 불가: 상한을 넘었다, 이미 결제했는데 필수 사전 합의가 없었다, 규칙에 어긋나는 투자 주문이다, 기준에 못 미친다
2. 판정 절차 6번 끝에: "앞으로 할 행동인데 합의 여부가 질문에 나오지 않으면 '추가 확인'이 아니라 '사전 합의 필요'다"
3. V3판만: 'V2 합본' → 'V3 합본', '표 54~57' → '표 53~57'

절차 3번("개정 항목인데 날짜가 없어 판본에 따라 결론이 갈리면 추가 확인")은 그대로 두었습니다.

### 바로 돌리기: 실험ID는 정해져 있습니다

```bash
bash 평가_실행.sh qwen3-32b-awq Qwen/Qwen3-32B-AWQ 모두                      # 기준선 p3 (V2 + V3)
bash 평가_실행.sh qwen3-4b-instruct-2507 Qwen/Qwen3-4B-Instruct-2507 모두    # 학생 p3 (V2 + V3)
```

| 조건 | 만드는 실험ID | 내용 |
|---|---|---|
| `v2p3` | T2MRp3_150, T2MRp3_N90 | V2 문서 + p3 |
| `v3p3` | T2MRp3_V3_150, _N90, _F30 | V3 문서 + p3_V3 (A의 V3 bge 검색이 없으면 이름이 `T2MRp3_V3bm_…`) |
| `모두` | 위 둘 다 | |
| `v2p2` | T2MR_150, T2MR_N90 | 11/19 기존 조건 그대로 |
| `점검` | CHECK5 | ④ 작동 확인 |

---

## ③ 최종 제출 파일 만드는 법

> B는 공개 100(평가 키트)을 봉인으로 정해 열지 않았습니다. 그래서 **운영진 제출 서식을 모릅니다.** C가 키트의 서식을 보고 `to_submission.py` 맨 위 `COLUMNS`의 열 이름과 순서를 맞춰 주세요. 지금 기본 열은 문항번호 · 판정 · 답 · 적용_판본 · 기준일 · 근거 · 다음_행동 · 확인_질문입니다.

1. **A**: 공개 100을 V3 문서로 검색 내보내기 (코랩, bge 켬)

```bash
python -m scripts.export_retrieval --questions <키트>/01_공개_배포평가셋_100문항.jsonl --name public100 --doc V3
```

2. 받은 파일 `retrieval_public100_V3.jsonl`을 `ver5/` 폴더에 올리고:

```bash
python make_questions.py retrieval_public100_V3.jsonl data/공개_100.jsonl
python convert_a.py retrieval_public100_V3.jsonl data/검색결과_공개_100_V3.jsonl data/판본_공개_100_V3.jsonl data/공개_100.jsonl
```

3. 최종 모델로 돌리기 (예: gemma4-12b-qat. 모델ID · 저장소만 바꾸면 됨)

```bash
python -c "from huggingface_hub import snapshot_download as s; s('google/gemma-4-12B-it-qat-w4a16-ct', local_dir='/workspace/models/gemma4-12b-qat')"
python run_all.py --questions data/공개_100.jsonl --retrieval data/검색결과_공개_100_V3.jsonl --retrieval-ver a-g3-bge \
  --exp FINAL_PUB100 --qset public100 --out results/results_FINAL_PUB100.jsonl --models gemma4-12b-qat \
  --examples 예시_p3.json --system-file p3_V3.txt --options options_v3.json \
  --use-memo --version-file data/판본_공개_100_V3.jsonl --concurrency 8 --max-tokens 2048
```

4. 제출 CSV로 펼치기 (마지막 인자로 질문 파일을 주면 R3까지 적용)

```bash
python to_submission.py results/results_FINAL_PUB100.jsonl gemma4-12b-qat 제출_공개100.csv data/공개_100.jsonl
```

- 공개 100은 **최종 조합을 확정한 뒤 한 번만** 돌립니다
- p2로 낼 거면 3번에서 `--examples 예시_p2.json`만 두고 `--system-file`을 빼세요. 단, p2 지시문은 'V2 합본' 기준입니다

---

## ④ 작동 확인 (문항 5개, 약 3분 + 모델 받기)

```bash
bash 평가_실행.sh qwen3-4b-instruct-2507 Qwen/Qwen3-4B-Instruct-2507 점검
```

- p2 지시문 · V2 검색 · 판본 고정으로 5문항을 돌립니다
- 아래 5문항은 11/19 같은 조건에서 12B · 4B · 32B · 31B가 **모두 맞힌** 문항입니다

| 문항 | 기대 상태 | 기대 판본 |
|---|---|---|
| D041 | 가능 | V2 |
| D135 | 가능 | V1 |
| D094 | 추가 확인 | 미정 |
| D006 | 규정에 없음 | 해당 없음 |
| D089 | 공동 처리 불가 | V2 |

- 판단 기준
  - **판본 5/5가 아니면 환경 문제**입니다 (판본은 코드가 고정하므로 반드시 같아야 함)
  - 상태가 4/5 이상이면 정상입니다
  - 요약은 `results/평가_요약.md`의 CHECK5 줄에 나옵니다. 문항별 출력은 `results/results_CHECK5.jsonl`

---

## ⑤ LoRA 학습

- **돌려 본 스크립트는 없습니다.** 계획한 설정으로 초안 2개를 넣어 두었습니다 (`train_lora.py`, `merge_lora.py`, GPU에서 미검증)
- 설정

| 항목 | 값 |
|---|---|
| 라이브러리 | transformers + peft (Trainer 사용) |
| 학생 | qwen3-4b-instruct-2507 (bf16 원본) |
| LoRA | 랭크 16, alpha 32, dropout 0.05, 대상 = 모든 선형층(q k v o gate up down) |
| 학습률 · 에폭 | 1e-4, 2에폭, 코사인, 워밍업 3% |
| 배치 | 1 × 누적 8, bf16, 그라디언트 체크포인팅 |
| 길이 | 최대 8192토큰, 넘는 문항은 뺌 |
| 입력 | 지시문(p3) + 검색 조각 + 판정 메모 + 판본 줄 + 질문 (실제 서빙과 똑같이 run_all 함수로 만듦) |
| 정답 | 교사 JSON 원문 그대로. 손실은 정답 토큰에만 |

순서:

```bash
# 1) 학습 데이터: C가 고른 문항번호 목록(한 줄에 하나)을 C가_고른_문항.txt로 저장한 뒤
python make_sft.py results/results_TEACHp3_860_V3.jsonl gemma4-31b-qat data/문항_train860_V3.jsonl \
  data/검색결과_train860_V3.jsonl data/판본_train860_V3.jsonl p3_V3.txt data/sft_V3.jsonl C가_고른_문항.txt

# 2) peft는 따로 둔 가상환경에 설치 (vLLM이 쓰는 transformers를 건드리지 않게)
python -m venv --system-site-packages /workspace/venv_lora
. /workspace/venv_lora/bin/activate && pip install peft

# 3) 학습 → 합치기 (같은 가상환경 안에서)
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-4B-Instruct-2507', local_dir='/workspace/models/qwen3-4b-instruct-2507')"
python train_lora.py data/sft_V3.jsonl /workspace/models/qwen3-4b-instruct-2507 /workspace/lora_qwen3-4b
python merge_lora.py /workspace/models/qwen3-4b-instruct-2507 /workspace/lora_qwen3-4b /workspace/models/qwen3-4b-lora
deactivate

# 4) 평가: 저장소 자리에 - 를 주면 받지 않고 합친 폴더를 그대로 씀
bash 평가_실행.sh qwen3-4b-lora - 모두
```

- V2·V3 교사 답을 같이 쓰려면 1)을 두 번 돌려 두 파일을 `cat data/sft_V2.jsonl data/sft_V3.jsonl > data/sft_all.jsonl`로 합칩니다
- vLLM에서 LoRA를 합치지 않고 띄우는 방법(`--enable-lora --lora-modules`)도 있습니다. 하지만 run_all이 그 방식을 지원하지 않아서 합치는 쪽을 권합니다

---

## 실행 방법

### 환경

```bash
nvidia-smi
python -c "import vllm, torch, transformers; print('vLLM', vllm.__version__, '| torch', torch.__version__, '| CUDA', torch.version.cuda, '| transformers', transformers.__version__)"
pip freeze > 환경_pip_freeze.txt
```

- vLLM은 **직접 띄우지 않습니다.** run_all.py가 모델마다 아래처럼 띄우고, 다 돌면 끕니다

```
vllm serve /workspace/models/<모델ID> --served-model-name <모델ID> --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 8 --seed 0 --generation-config vllm [모델별 인자]
```

- 서버 로그는 `results/logs/<모델ID>.log`에 쌓입니다

### 모델별 예외 설정 (`run_all.py`의 `MODELS`)

앞 인자로 안 뜨면 다음 인자로 다시 띄웁니다.

| 모델 | 인자 |
|---|---|
| gemma4-12b-qat · gemma4-e4b · gemma4-e2b · qwen3.5-4b · qwen3.5-2b | `--language-model-only` (그림·소리 인코더 빼기) → 안 되면 없이 |
| qwen3.8-27b-awq | `--language-model-only` → 없이 |
| qwen3-32b-fp8 | 없음 → `--max-model-len 24576 --gpu-memory-utilization 0.95` (11/19에 이 두 번째로만 떴음) |
| qwen3.8-27b-fp8 · gemma4-31b-fp8 | `--language-model-only` → + 문맥 24576 · GPU 0.95 |
| exaone 계열 · kanana2-3b | `--trust-remote-code` |
| kanana2-1.3b | `--trust-remote-code` → + `--model-impl transformers` |

- 추론(생각) 모드는 요청마다 끕니다 (`enable_thinking: False`)

### 모델 받기

```bash
python -c "from huggingface_hub import snapshot_download as s; s('google/gemma-4-31B-it-qat-w4a16-ct', local_dir='/workspace/models/gemma4-31b-qat')"
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-4B-Instruct-2507', local_dir='/workspace/models/qwen3-4b-instruct-2507')"
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-32B-AWQ', local_dir='/workspace/models/qwen3-32b-awq')"
```

| 모델 | 역할 | 크기(가중치) | 받는 시간 |
|---|---|---|---|
| gemma4-31b-qat | 교사 | 약 20GB | 약 5~15분 |
| qwen3-4b-instruct-2507 | 학생 | 약 8GB | 약 2~5분 |
| qwen3-32b-awq | 기준선 | 약 19GB | 약 5~15분 |

받는 시간은 어림값입니다. 볼륨이 100GB라서 큰 모델 3개를 동시에 두지 말고, 끝나면 `rm -rf /workspace/models/<모델ID>`로 지웁니다.

### run_all.py 사용법

```bash
python run_all.py --questions <질문.jsonl> --retrieval <검색결과.jsonl> --retrieval-ver <이름표> \
  --exp <실험ID> --qset <질문세트 이름> --out results/results_<실험ID>.jsonl --models <모델ID> \
  --examples <예시.json> [--system-file <지시문.txt>] [--options options_v3.json] \
  --use-memo --version-file <판본.jsonl> --concurrency 8 [--limit N] [--max-tokens 2048]
```

- 바꾸는 곳: 실험ID = `--exp`와 `--out`, 모델 = `--models`(쉼표로 여러 개), 입력 = `--questions`·`--retrieval`·`--version-file`, 지시문 = `--system-file`(없으면 p2)
- 같은 실험ID · 모델 · 문항이 이미 성공해 있으면 건너뜁니다. 설정을 바꿨으면 **실험ID를 새로** 주세요
- 응답 시간은 `--concurrency 1`일 때만 지표로 씁니다

예시 3개:

```bash
# 1) 기준선 p3 (V2)
python run_all.py --questions data/개발문항_150.jsonl --retrieval data/검색결과_T2_150.jsonl --retrieval-ver a-g2.2-bge \
  --exp T2MRp3_150 --qset dev150v2 --out results/results_T2MRp3_150.jsonl --models qwen3-32b-awq \
  --examples 예시_p3.json --system-file p3_V2.txt --use-memo --version-file data/판본_A규칙_150.jsonl --concurrency 8
# 2) 학생 p3 (V2): 1)에서 --models만 qwen3-4b-instruct-2507
# 3) 교사로 학습 질문 답 만들기: teach.sh가 V2·V3를 다 합니다 (아래 '교사 답 생성')
```

### 실행 설정이 코드 어디에 있나 (`run_all.py`)

| 설정 | 위치 |
|---|---|
| temperature 0 · seed 0 · JSON 형식 강제(json_schema strict) · 생각 모드 끔 | `ask()`의 `body` |
| 최대 출력 토큰 1536 | `--max-tokens` 기본값 (형식 30은 2048) |
| 출력 칸과 선택지 | `build_schema()` + `options_v2.json` / `options_v3.json` |
| 판본 칸을 A 규칙 값 하나로 제한 | `lock_version()` (`--version-file`을 주면 켜짐) |
| 서버 설정(문맥 32768, GPU 0.90 등) | `COMMON_ARGS` |
| p2 지시문 | `SYSTEM_PROMPT` (p3는 파일) |

### V3로 돌리는 법

1. **A의 V3 검색 파일** (`retrieval_dev150_V3.jsonl` 등, bge 켬)을 이 폴더에 올리고:

```bash
python convert_a.py retrieval_dev150_V3.jsonl data/검색결과_T2v3_150.jsonl data/판본_A규칙v3_150.jsonl data/개발문항_150.jsonl
python convert_a.py retrieval_n90_V3.jsonl data/검색결과_T2v3_잡음_90.jsonl data/판본_A규칙v3_잡음_90.jsonl data/잡음문항_90.jsonl
python convert_a.py retrieval_f30_V3.jsonl data/검색결과_T2v3_형식_30.jsonl data/판본_A규칙v3_형식_30.jsonl data/형식문항_30.jsonl
```

   - 이 파일들이 있으면 `평가_실행.sh … v3p3`가 자동으로 bge판을 씁니다 (실험ID `T2MRp3_V3_…`)
   - 지금 들어 있는 것은 B가 BM25만으로 만든 임시판입니다 (실험ID `T2MRp3_V3bm_…`)
   - convert_a.py가 "질문 대조: 다른 문항 N개"를 출력하면, 검색에 쓴 질문과 우리 질문 파일이 다른 것입니다

2. **V3 정답지 4개**를 `data/`에 두고 V3 실험만 채점:

```bash
python 채점.py --gold data/개발정답_V3_D001-D150.csv data/잡음변형_V3_N001-N090.csv data/형식변형_V3_F01-F30.csv --exp V3 --out results/평가_요약_V3.md
```

   - 정답 CSV의 칸 이름이 지금 정답지와 같다고 가정했습니다 (`문항번호`, `질문`, `정답_상태`, `적용_판본`). 다르면 칸 이름만 맞춰 주세요
   - V3 쌍 16문항(`V3쌍문항_V01-V16.csv`)은 질문 파일이 필요합니다. A가 V2판·V3판 검색을 둘 다 내보내면 아래로 질문 파일을 만들고 위와 같은 방식으로 돌립니다

```bash
python make_questions.py retrieval_pair16_V3.jsonl data/V3쌍문항_16.jsonl
```

### 채점 · 그림 · 응답 시간

```bash
python 채점.py                              # 빠른 요약 → results/평가_요약.md (상태 · 주지표 · R3 적용 · 판본 바뀐 문항 뺀 점수)
MPLBACKEND=Agg python 모델비교_시각화.py      # 그림 6개 → /workspace/DLthon02/ver4/results/그림_비교/ (ver3·ver4 결과를 읽음)
```

- 정식 채점(유형별 표 · 혼동표)은 팟의 `ver4/score_cell.py`입니다. `run_all.ipynb`의 채점 칸 맨 위에서 `RESULT_FILES`, `ANSWERS`, `MAIN_EXP`만 바꿉니다
- 응답 시간 (동시 1, 앞 30문항):

```bash
python run_all.py --questions data/개발문항_150.jsonl --retrieval data/검색결과_T2_150.jsonl --retrieval-ver a-g2.2-bge \
  --exp LAT_150 --qset dev150v2 --out results/results_LAT_150.jsonl --models <모델ID> --examples 예시_p2.json \
  --use-memo --version-file data/판본_A규칙_150.jsonl --concurrency 1 --limit 30
```

- 메모리: 결과 옆 `results/results_<실험ID>_models.jsonl`의 `가중치_메모리_GiB`(vLLM 로그에서 읽은 값)와 `파일_크기_GB`

### 교사 답 생성

A가 보낼 `retrieval_train860_V2_g32.jsonl`, `retrieval_train860_V3_g32.jsonl`을 이 폴더에 올리고:

```bash
bash teach.sh retrieval_train860_V2_g32.jsonl retrieval_train860_V3_g32.jsonl
```

- 기본 교사는 gemma4-31b-qat입니다. 한 번 받아서 V2 · V3를 둘 다 돌리고 지웁니다
- 결과: `results/results_TEACHp3_860_V2.jsonl`, `results/results_TEACHp3_860_V3.jsonl`. 한 줄에 `출력_원문`(교사 JSON 원본 전체)이 있습니다
- 예전 `retrieval_train800_V3.jsonl`은 쓰지 않습니다
- 860 × 2라 오래 걸립니다 (어림 2~3시간). 끊기면 같은 명령을 다시 치면 이어서 합니다

---

## 지금 상태

### 끝난 것

- V2 문서, 모델 19개 × O1p_150 · T2MR_150 · T2MR_N90 (오류 0)
- 소형 4개(qwen3-4b-instruct-2507 · qwen3.5-4b · gemma4-e4b · kanana2-3b) × 12조건
- 응답 시간(LAT_150) · 형식 30(O1p_F30): 소형 4개 + qwen3-32b-awq · qwen3-32b-fp8
- 채점 CSV, 그림 6개, R3 후처리 시뮬레이션
- p3 지시문 확정본, 평가 · 교사 · 제출 · 증류 스크립트 (이 압축)

### 하다 만 것

- V3 검색: B가 BM25만으로 임시판을 만듦. A의 bge판으로 바꿔야 함
- 가지치기(`prune_layers.py`, 층 빼기) · LoRA(`train_lora.py`): 초안만 있고 GPU에서 미검증

### 안 한 것

- p3로 돌린 결과 없음 (기준선 · 학생 모두)
- V3 문서로 모델을 돌린 결과 없음
- 교사 답, LoRA 학습, 공개 100
- gemma4-12b-qat 응답 시간

### 결과 파일 위치 (팟)

| 위치 | 내용 |
|---|---|
| `/workspace/DLthon02/ver3/results/` | 11/19 낮: 소형 4개 × 12조건 |
| `/workspace/DLthon02/ver4/results/` | 11/19 밤: 19개 모델 × 3조건, LAT_150, O1p_F30 |
| `/workspace/DLthon02/ver5/results/` | 이 묶음으로 돌린 것 (p3 · V3 · 교사 답) |
| `ver4/results/채점/` | 채점 CSV |
| `ver4/results/그림_비교/` | 그림 6개 (PNG) |

### 현재 숫자 (V2, T2MR_150, /150)

| 모델 | GiB | 원점수 | R3 적용 |
|---|---|---|---|
| qwen3-32b-awq (기준선) | 18.24 | 79 | 91 |
| gemma4-12b-qat | 8.29 | 91 | 98 |
| qwen3-4b-instruct-2507 | 7.64 | 74 | 85 |
| gemma4-e4b | 14.17 | 77 | 83 |
| gemma4-31b-qat (교사) | 19.77 | 133 | — |

### 알려진 문제

- **1536토큰 잘림**은 1~2B 모델이 같은 말을 반복할 때만 생깁니다 (kanana2-1.3b 29 · exaone4.0-1.2b 24 · qwen3.5-2b 11). 길이가 모자란 게 아니라 반복이라, 토큰을 늘려도 해결되지 않습니다
- **FP8이 A40에서 W8A16으로 돔**: A40에는 FP8 연산이 없어 가중치만 8비트(FP8 Marlin)입니다. 그래서 4비트보다 느립니다 (qwen3-32b-fp8 30.4초 vs awq 18.7초)
- qwen3-32b-fp8은 문맥 24576 · GPU 0.95로만 떴는데, 결과 행의 설정 칸에는 32768로 적혀 있습니다 (기록 불일치)
- qwen3-8b · qwen3-14b · kanana2-1.3b는 '없음'이 정답일 때 '모름'을 자주 씁니다

---

## 1분이면 되는 정리

1. **잡음 90 질문을 원래 문구판으로 교체**: ver5에는 **이미 반영**했습니다 (13문항: N011 N017 N034~N039 N052~N054 N077 N086). A의 잡음 검색(V2 bge)과 질문이 다시 똑같아졌고, V3 임시 검색도 원래 문구로 다시 만들었습니다. 정답지의 질문 글자도 원래 문구로 맞췄습니다(정답은 그대로)

2. **F09 적용판본 → '미정'**: 이 압축의 `data/개발정답_150_N90_F30.csv`에는 **이미 반영**했습니다. 팟 ver4의 정답지도 바꾸려면:

```bash
python -c "
import csv
p='/workspace/DLthon02/ver4/data/개발정답_150_N90_F30.csv'
rows=list(csv.DictReader(open(p,encoding='utf-8-sig'))); f=list(rows[0])
for r in rows:
    if r['문항번호']=='F09': r['적용_판본']='미정'
w=csv.DictWriter(open(p,'w',encoding='utf-8-sig',newline=''),fieldnames=f); w.writeheader(); w.writerows(rows); print('F09 → 미정')"
```

3. **qwen3-4b p3 한 번** (p2 74점과 비교): `bash 평가_실행.sh qwen3-4b-instruct-2507 Qwen/Qwen3-4B-Instruct-2507 v2p3` → 요약의 `T2MRp3_150` 줄

---

## 넘겨주는 파일 (`ver5_1120.zip`)

| 파일 | 내용 |
|---|---|
| `run_all.py` | 실행기 (`--system-file` 추가, 증류 모델 `qwen3-4b-lora` 등록) |
| `p3_V2.txt`, `p3_V3.txt`, `예시_p3.json`, `예시_p2.json` | 지시문 · 예시 |
| `options_v2.json`, `options_v3.json` | 출력 칸 선택지 (V3: 판본 V3, 개정 항목 11개, 기준일 종류 3개 추가) |
| `평가_실행.sh` | 받기 → 돌리기 → 채점 → 지우기 |
| `teach.sh`, `make_questions.py`, `convert_a.py` | 교사 답, A 파일 변환 |
| `채점.py`, `to_submission.py` | 빠른 채점, 제출 CSV |
| `make_sft.py`, `train_lora.py`, `merge_lora.py` | 증류 (뒤의 두 개는 미검증 초안) |
| `prune_layers.py` | 층 빼기 가지치기 (미검증 초안) |
| `모델비교_시각화.py` | 그림 6개 |
| `data/` | 질문(잡음 90은 원래 문구판) · 검색 · 판본 · 정답지(F09 수정), 점검 5문항 |
| 11/19 `B_최종_v1`의 파일 전부 | 오라클 검색 결과, `run_all.ipynb`, `score_cell.py`, `make_t2.py`, `make_version.py`, `실행표_1119.md` 등 그대로 |

- 원본 출력(results*.jsonl) · 채점 CSV · 그림 PNG는 용량 때문에 압축에 넣지 않았습니다. 위 '결과 파일 위치'의 팟 폴더에 있습니다

---

## B 의견 (한 줄씩)

- **p3가 p2보다 나쁘면**: 개발 150과 잡음 90 **둘 다** 나쁘면 p2를 씁니다. 하나만 나쁘면 잡음 90 쪽을 믿습니다(덜 과적합). 차이가 2문항 이내면 우연 범위라, 정의가 분명한 p3를 유지합니다. 어느 쪽이든 기준선도 같은 지시문으로 비교합니다
- **증류가 실패하면**: **gemma4-12b-qat + T2MR + R3** (V2 개발 150에서 98 vs 기준선 91, 무게 45%). qwen3-4b는 R3를 써도 합격선(88)보다 3문항 낮습니다
- **월요일 발표 B 파트 핵심 문장**
  - 모델 19개를 같은 검색 · 지시문 · 설정으로 비교했고, 4비트 12B(8.3GiB)가 기준선 32B(18.2GiB)보다 더 맞혔다
  - 같은 모델이면 4비트가 8비트보다 같거나 나았다. A40에서는 8비트가 오히려 느렸다
  - 작은 모델은 금액 · 합의 여부를 거의 다 맞게 뽑았지만, 그다음 상태를 고르는 데서 틀렸다. 그래서 상태 고르기의 한 부분을 코드(R3)로 넘기자 기준선 포함 전 모델이 올랐다
  - 1~2B는 같은 말을 반복하다 잘려서 바닥 수준이었다. 경량화의 하한선이 여기다
- **이건 꼭**
  - 지시문 · 예시를 바꾸면 **기준선부터** 같은 조건으로 다시 돌리고, 그다음 학생을 돌립니다
  - 설정이 바뀌면 **실험ID를 새로** 씁니다 (같은 ID면 '이미 끝남'으로 건너뛰거나 섞입니다)
  - 공개 100은 최종 조합을 확정한 뒤 **한 번만**
- **이건 하지 마**
  - vLLM을 두 개 동시에 띄우기 (run_all이 포트 8000을 확인하고 멈춤, GPU 메모리 0.90을 씀). 교사 답을 만드는 동안 다른 GPU 작업 금지
  - 팟 **Terminate**
  - (A) `add_version --write` 다시 돌리기 (V3 손 입력이 지워짐)

## 넘겨받는 순서 (팀 제안 그대로)

| 순위 | 할 일 | 명령 |
|---|---|---|
| 0 | 환경 확인 | `bash 평가_실행.sh qwen3-4b-instruct-2507 Qwen/Qwen3-4B-Instruct-2507 점검` |
| 1 | p3로 기준선 · 학생 재측정 (V2 · V3) | `bash 평가_실행.sh qwen3-32b-awq Qwen/Qwen3-32B-AWQ 모두`, 학생도 같은 방식 |
| 2 | 교사 답 860 × V2 · V3 | `bash teach.sh <V2 파일> <V3 파일>` |
| 3 | C가 교사 답 거르기 | 결과 jsonl의 `출력_원문` |
| 4 | 학생 LoRA → 평가 (C) | ⑤ 순서 |
| 5 | 계산기 등 코드 개입 (B) | 'B 트랙' |
| 6 | 경량화 (B) | 'B 트랙', 아니면 발표의 '다음 과제' |

## B 트랙: 경량 모델 끌어올리기

**진단**: 작은 모델(4B 이상)은 금액 합산(20~24/24)과 합의 여부(51~52/52)를 거의 다 맞게 뽑습니다. 틀린 상태 대부분은 뽑기가 아니라 **상태를 고르는 단계**에서 나옵니다.

**방향**: 모델이 뽑고(추출) → 코드가 계산 · 판정하고 → 모델은 설명만. 기계적으로 정할 수 있는 판정을 코드로 넘깁니다.

| 방법 | 무엇을 하나 | 상태 | 효과 · 근거 |
|---|---|---|---|
| R3 후처리 | '추가 확인'인데 빠진 정보가 합의 · 동의뿐이고 앞으로의 일이면 → '사전 합의 필요' | **완료** (p3 절차 6번에도 반영) | 개발 150: 4B 74→85, 12B 91→98, 기준선 79→91. 망친 문항 모델당 1개 이하 |
| 계산 힌트 | 코드가 합산액 · 상한 비교 · 초과 여부를 계산해 판정 메모에 넣음 | 다음 차례 | 작은 모델이 계산을 안 해도 됨 |
| R1 다시 만들기 | '가능'인데 합산액 > 상한이면 '공동 처리 불가'. 모델이 뽑은 상한 대신 코드가 표에서 읽은 상한을 씀 | 계획 | 처음 R1은 좋은 답을 망쳐서 뺐음 |
| 규칙 사전 | 규칙 틀 약 9개 + 수치는 규약 표에서 읽기. V3로 바뀌어도 표만 다시 읽으면 됨 | 설계 | 정확도 기대가 가장 큼 |
| 슬롯 채우기 | 판정에 필요한 칸(날짜 · 금액 · 합의 · 이미 했는지)을 근거 문장과 함께 뽑고, 빠진 칸이 결론을 바꿀 때만 되묻기 | 설계 | 무조건 되묻기 금지 |
| 잘림 방지 | 출력 형식에 칸별 최대 개수 · 길이 제한 | 설계 | 1~2B의 반복 잘림 |
| 양자화 | 4비트 · 8비트 · 16비트 비교 | **완료** | 4비트가 8비트보다 같거나 나음. 12B 4비트가 가장 작은 합격 모델 |
| 구조적 가지치기 | 층 빼기 (ShortGPT 방식) | 초안 (`prune_layers.py`, 미검증) | 파일 · 메모리가 실제로 줄어듦 |

- 권하는 순서: R3(완료) → 계산 힌트 → R1 다시 만들기 → 규칙 사전 일부
- 발표 전에 현실적인 것은 **R3 + 계산 힌트**까지입니다. 나머지는 '다음 단계'로 보여 줍니다
- 비구조적 가지치기(가중치 일부를 0으로)는 파일 크기와 메모리가 줄지 않습니다. 2:4 희소화는 vLLM 커널이 Hopper 위주라 A40에서 이득이 불확실합니다
- 층 빼기는 재학습 없이 하면 점수가 크게 떨어질 가능성이 높습니다. 후보가 아니라 '경량화의 대가'를 보여 주는 자료로 씁니다
- 주의: R3와 p3는 개발 150 오답을 보고 만들어 그 점수는 낙관적입니다. 코드 개입은 기준선에도 똑같이 적용해 비교합니다(합격선 88). 규칙 사전을 극한까지 하면 "코드가 판정한다"는 비판이 나올 수 있으니, 발표에서 모델이 하는 일과 코드가 하는 일을 나눠 보여 줍니다

## 수정 기록

- 2026-11-20 15시 작성 (A·C 요청 항목에 맞춰 정리, p3 확정본 · 실행 묶음 포함)
- 2026-11-20 15시 폴더를 ver5로 바꾸고 맨 앞에 사용법 표 · 새 세트마다 verN 규칙 추가, 팟 결제 · 커밋 해시 칸 삭제
- 2026-11-20 15:12 역할 분담(C 증류 · A 검색 파일 · B 경량 모델 끌어올리기)과 'B 트랙' 절 추가
- 2026-11-20 15:20 ver5를 B_최종_v1의 확장판으로 정리: v1 파일 전부 포함, 잡음 90을 원래 문구판으로 교체(정답지 질문 글자 · V3 임시 검색도 함께)
