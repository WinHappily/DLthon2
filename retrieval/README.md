# retrieval — 문서 · 검색 · 판본 규칙 (A · 권해나)

| 경로 | 역할 |
| --- | --- |
| `rag/extract.py` | PDF에서 본문 · 표 추출, 쪽 번호 보존 |
| `rag/index.py` | 쪽 단위 청크 + 꼬리표, 표 행 보조 청크, bge-m3 색인 |
| `rag/search.py` | 검색 · 재순위 · 조항 참조 따라가기 |
| `rag/version.py` | 판본 규칙(개정 항목 → 기준일 종류 → 날짜 → 시행 구간) |
| `scripts/` | 추출 점검, 판본 추가, V2 · V3 차이 비교, 한 문항 질의 |
| `eval/` | 검색 품질(정답 조항 상위 k 포함률) · 판본 규칙 정확도 평가 |
| `notebooks/role_A_colab.ipynb` | Colab에서 색인을 만들고 평가 · 학습용 검색 결과 파일을 내보내는 노트북 |

이 폴더는 팀 공유 시점의 A 저장소 사본이다. 원래 README의 경로는 이 폴더 기준이다(`data/V2/`, `eval/`). 규약집 PDF는 `pdf/rulebook_V2.pdf`처럼 두라고 되어 있는데, 배포 원본은 `../regulations/`에 있다. 최종 판본 규칙 g3.3과 V3 색인 산출물은 Colab에서 만들었고, 그 결과(검색 결과 파일)는 `../data/retrieval_V3/`에 있다.

---

아래는 A 저장소의 원래 README.

# 신혼생활 운영위원회 규약집 검색 (DLThon2 · 역할 A)

규약집 PDF를 검색 가능한 형태로 만들고, 질문마다 **맞는 판본의 맞는 조항 · 표 · 쪽**을 찾아 주는 코드입니다.
생성(LLM)은 들어 있지 않습니다. `retrieve()`의 결과를 서빙(B) · 프롬프트(C)가 받아 씁니다.

## 코랩에서 실행 (가장 쉬운 길)

1. 구글 드라이브에 `MyDrive/dlthon2/pdf/rulebook_V2.pdf`를 둔다 (PDF는 깃에 올리지 않는다)
2. [notebooks/role_A_colab.ipynb](notebooks/role_A_colab.ipynb)를 코랩에서 연다 (GitHub에서 열기 또는 드라이브 업로드)
3. 첫 셀의 `REPO_URL`에 이 저장소 주소를 넣는다. 비공개 저장소면 코랩 '보안 비밀'에 `GH_TOKEN`
   (저장소 없이 하려면 `REPO_URL=""`로 두고 이 폴더를 zip으로 묶어 `MyDrive/dlthon2/`에)
4. 런타임 → GPU(T4) → 모두 실행. V3가 나오면 `rulebook_V3.pdf`를 같은 폴더에 넣고 5절만

## RunPod 새 팟에서 처음부터 (B · 서빙 쪽)

`data/V2/`(추출 · 조각 · 조항 지도)는 저장소에 들어 있다. 벡터(`emb.npy`)만 GPU에서 한 번 만든다.

```bash
cd /workspace
git clone https://github.com/<아이디>/dlthon2-rag.git     # 비공개면 토큰 또는 협업자 초대
cd dlthon2-rag
python -m venv --system-site-packages .venv && source .venv/bin/activate   # 팟의 torch를 그대로 씀 (16G 재설치 방지)
pip install --no-cache-dir -r requirements.txt sentence-transformers
export HF_HOME=/workspace/.cache/huggingface
python -m rag.index --doc V2 --embed BAAI/bge-m3      # bge-m3 약 2GB 받고 벡터 생성
python -m eval.eval_retrieval --doc V2                # 확인 : 검색+참조+판본 줄 version_acc 16/16
```

BM25만 쓸 거면 `--embed`를 빼고 `sentence-transformers`도 안 깔아도 된다. PDF는 `rag.extract`를 다시 돌리거나 `check_extract`로 꼬리말을 대조할 때만 필요하다(`pdf/rulebook_V2.pdf`, git 제외).

## 터미널에서 실행 (로컬 · RunPod)

```bash
python -m venv .venv && .venv/Scripts/activate          # 리눅스 · RunPod : source .venv/bin/activate
pip install -r requirements.txt

python -m rag.extract --doc V2            # A-1 · A-2 · A-3  PDF → data/V2/pages.jsonl · rows.jsonl · article_map.json · report.txt
python -m rag.index   --doc V2            # A-4              조각 402개(쪽 60 + 표 행 342). BM25는 검색할 때 바로 만든다
python -m rag.search  "어머님 생신 식사와 선물을 다른 카드로 결제했어" --doc V2
python -m rag.version "공용 밥솥을 35만 원에 9월 30일 주문 확정했어"
python -m eval.eval_retrieval --doc V2    # A-5              results/retrieval_V2_<시각>.jsonl
python -m eval.eval_version   --doc V2    # A-6              판본 규칙만 채점 (30문항 + 처음 보는 15문항)
```

의미 검색(벡터)을 더하려면 GPU에서 `pip install sentence-transformers` 후
`python -m rag.index --doc V2 --embed BAAI/bge-m3`. `emb.npy`가 있으면 `rag.search`가 자동으로 BM25와 RRF로 합칩니다.

> Windows 로컬에서 최신 numpy 휠이 "애플리케이션 제어 정책"으로 막히면 `pip install "numpy==2.2.6"`.

## 첫날 시험 두 가지

### (A) PDF 추출 시험

```bash
python -m scripts.check_extract --doc V2                    # 기본 17 · 41 · 54쪽
python -m scripts.check_extract --doc V2 --pages 3 17 59    # 쪽을 바꿔서
```

쪽마다 자동 검사(쪽 번호 꼬리말 대조 · 조항 · 본문 길이 · 표 번호 = 쪽 번호 · 표 행 · 칸 안 줄바꿈 복원 · 알려진 행)를 하고,
프롬프트에 들어갈 본문과 표 행 문장을 그대로 보여 준다. 결과는 `results/check_extract_V2_<시각>.txt`.

### (A · B) 추출한 쪽으로 질문 하나에 답 받기

터미널 1 — 서버 (venv 밖, 팟에 깔린 vLLM). 모델은 팀이 정한 것으로 바꾼다. A40 48GB에서 큰 모델 + 작은 모델을 같이 띄울 때 :

```bash
BIG=/workspace/models/<큰 모델 폴더>        # 또는 허깅페이스 이름
vllm serve $BIG --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.60
```

터미널 2 — (선택) 작은 모델 :

```bash
SMALL=/workspace/models/<작은 모델 폴더>
vllm serve $SMALL --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.25
```

터미널 3 — 질문 :

```bash
cd /workspace/dlthon2-rag && source .venv/bin/activate
python -m scripts.ask_once --dry-run                                         # 프롬프트만 확인
python -m scripts.ask_once --model $BIG   --base-url http://localhost:8000/v1   # --model 은 vllm serve 에 준 값 그대로
python -m scripts.ask_once --model $SMALL --base-url http://localhost:8001/v1
python -m scripts.ask_once --pages 41 42 54 14 4 --model ...                 # 쪽을 손으로 지정
python -m scripts.ask_once --question "다른 질문" --model ...
```

기본 질문은 표 59의 2행(정답 : 사전 합의 필요 · V2 · 22만 원 합산 · 제3조 · 제23조 · 제79조 · 표 41).
답 · 시간 · 토큰이 `results/ask_once.jsonl`에 쌓인다. 작은 모델은 완료 기준에 넣지 않는 '가능성 가늠'용.

## 업무와 파일

| 번호 | 업무 | 파일 | 하는 일 |
|---|---|---|---|
| A-1 | PDF 추출 | `rag/extract.py` | 쪽마다 본문과 표를 따로 뽑는다(표 영역은 본문에서 빼고 `pdfplumber` 표 인식으로). 머리말 · 꼬리말 제거. 칸 안 줄바꿈 복원(`join_wrapped`) |
| A-2 | 표 처리 | `rag/extract.py` | 표의 각 행 → `표 41 가족행사별 공동비 상한 V2, 행사: …, 행사당 상한: 22만 원, …` 문장(`rows.jsonl`) |
| A-3 | 청킹 | `rag/extract.py` | **쪽 = 조각**. 조각마다 판본 · 쪽 · 장 · 조항 · 표 · 나가는 참조 · 개정 여부(`pages.jsonl`) |
| A-4 | 색인 · 검색 | `rag/index.py`, `rag/search.py` | BM25(조항 · 표 번호는 통째로, 한글은 두 글자 조각) + (선택) 벡터 → RRF. 표 60 근거 색인의 검색어로 가산. 상위 쪽이 언급한 `제N조` · `표 NN`을 한 단계 따라가 붙임 |
| A-5 | 검색 품질 | `eval/eval_retrieval.py`, `eval/questions_seed.jsonl` | 정답 쪽 recall · all_hit · MRR, 판본 규칙 정확도 |
| A-6 | 기준일 · 판본 | `rag/version.py`, `config/versions.json` | 항목(표 07) → 기준일 종류 → 질문의 날짜 → 시행 구간(부칙 제1조). 경과조치 : 후속 결제 · 환불은 원거래 날짜, 전량 취소 후 새 주문 · 예약은 새 날짜. **문서마다 아는 판본이 다르다** |
| A-7 | V3 반영 | `scripts/add_version.py`, `scripts/diff_versions.py` | 새 PDF에서 시행일 · 표 07 항목 · 대조표 쪽을 자동으로 찾아 설정에 넣고, 조항 · 표 행 · 질문 판정 차이를 표로 |

## 현재 결과 (V2 · 씨앗 17문항 · BM25만 · k=5)

| 방식 | 정답 쪽 recall | 정답 쪽 모두 들어온 문항 | MRR | 판본 규칙 |
|---|---|---|---|---|
| 검색만 | 0.902 | 13/17 | 0.824 | — |
| 검색 + 참조 따라가기 + 판본 규칙 쪽 | **0.985** | **16/17** | 0.824 | **16/16** |

놓친 것 : T59-2의 4쪽(제3조 금액 단위 — '상한과 같은 금액 포함'). 평균 10.3쪽(약 6천 토큰)이 프롬프트에 들어가므로
B는 `--max-model-len 8192` 이상, 줄이려면 `retrieve(k=4, max_refs=2)`.

씨앗 질문 = 과제 예시 1 + 표 58(개정 전후 5쌍) 10 + 표 59(복합 사례) 6. 정답이 규약집에 그대로 있는 쉬운 문항이라 **C의 팀 질문 세트로 다시 잰다** :
`python -m eval.eval_retrieval --doc V2 --questions eval/questions_team.jsonl`

## 판본 규칙 v2 (11.18)

v1은 날짜가 여럿이면 가장 이른 날짜를 기준일로 썼다. 판본을 일부러 헷갈리게 만든 질문으로 재 보니 다음과 같았다.

| 질문 세트 | v1 | v2 |
|---|---|---|
| `eval/questions_version.jsonl` 30문항 (규칙을 고치면서 본 질문) | 19/30 | **30/30** |
| `eval/questions_version_holdout.jsonl` 15문항 (고친 뒤 처음 돌린 질문) | 8/15 | **14/15** → 1개 보완 후 15/15 |
| 씨앗 17문항의 판본 (`eval_retrieval`) | 16/16 | 16/16 |
| `eval/questions_version_team.jsonl` 팀 개발 세트 오답 · 변형 10문항 (v2.1) | – | **v2.1 10/10** |
| **`eval/questions_version_dev50.jsonl` 팀 개발 50문항** | **43/50** | v2 43/50 → **v2.1 50/50** |

v1이 틀린 이유와 v2에서 고친 것

| v1이 틀린 이유 | 예 | v2 |
|---|---|---|
| 날짜의 역할을 안 봄 | "10월 1일부터 2박, 9월 27일에 알렸어" → 고지일(9/27)로 V1 | 날짜 뒤 구절의 동사로 역할(결제 · 주문 · 예약 · 시작 · 종료 · 고지 · 문의 · 환불 ...)을 붙이고, 기준일 문구(표 07)가 받아들이는 역할의 날짜만 쓴다 |
| 경계 시각 | "9월 30일 24시 결제" → V1 | 24시 · 자정은 다음 날 00시, 경계 시각은 새 판본(표 53) |
| 기준일이 아닌 날짜만 있음 | "10월 5일이 생신이라 …" → V2로 단정 | 행사일만 있으면 첫 결제일을 묻는다(`추가 확인`, 표 05 '행사일과 첫 결제일 대조') |
| 개정 항목이 아니면 판본 표기 | "공동계좌 월 예산은?" | **v2.1** : 팀 정답 표기대로 `label = "해당 없음"`, 되묻지 않음(`applied_version`은 None, 본문은 `text_version` 판본) |
| 항목이 여럿 | 숙박(9/30)과 생신 결제(10/1)가 한 질문에 → 하나만 판정 | `by_item`에 항목마다 판정. 주 항목은 맞은 키워드가 많은 것 |
| 상대 날짜 | "10월 1일 아침 끝난 근무는 전날 밤 시작" | 전날 · 다음 날 · 같은 날을 앞 날짜로 계산 |

v2.1 (11.18, 팀 개발 50문항 중 판본 7문항 오답 대응)

| 팀 개발 세트 오답 | 원인 | 고친 곳 |
|---|---|---|
| Q018 "친구 모임 술자리" → 당일 방문으로 잡고 되묻기 | '모임'이 당일 방문 키워드 | 당일 방문은 집 안 방문만(`방문 · 놀러 · 초대 · 집에`), 숙박 · 다과 말이 있으면 제외 |
| Q019 "처가 명절 선물 · 음식" → 가족행사로 잡고 되묻기 | 표 54에서 바뀐 것은 **부모님 생신 상한뿐**인데 명절 · 혼례도 넣었음 | 가족행사 항목은 `생신 · 생일`만 |
| D018 · D019 "2박 손님 다과 · 소모품" → 손님 숙박으로 잡고 되묻기 | 표 56에서 바뀐 것은 2박 고지 · 합의뿐, 다과 · 소모품(표 45)은 판본 무관 | 항목에 `none`(이 말이 있으면 그 항목 아님) 추가 |
| D014 · D015 "국내 숙소 예약 확정 → 개인 사유 취소" → 항목 못 잡음 | '숙소' · '순취소비'가 키워드에 없음 | 여행 항목 키워드 추가 |
| D017 새 예약 표현 | 새 거래 표현이 좁음 | `새로 예약 · 다시 잡 · 별도 예약 · 다른 숙소 … 예약` 등 추가 |

| Q012 "10월 5일에 전부 취소하고 새 주문번호로 다시 주문 확정" → V1 (v2에서 생긴 회귀) | 그 날짜를 '취소'로만 봄 | 취소한 그날 새 주문 · 예약이면 그 날짜도 기준일 후보 |

개발 50문항의 정답 : 팀 정답 파일(`개발정답_50.csv`)의 `개정_항목` · `적용_판본`('미정' → 추가 확인) · `기준일`을 옮겼다(11.18).

원칙 : **개정 항목 = 표 54~57에서 V1과 V2 값이 실제로 다른 행**이다. 같은 항목 이름이라도 바뀌지 않은 하위 규정을 물으면 '해당 없음'이다.
V3 때도 `add_version`이 만든 새 항목은 이름 낱말만 키워드로 넣으므로, 대조표에서 **값이 바뀐 행**을 보고 `all` · `none`을 손본다.

`decide()` 결과에 더해진 것 : `dates`(날짜마다 역할과 구절, 디버깅용), `by_item`. 프롬프트(`ask_once`)에 판정 이유도 넣는다.

V3 대비 : 기준일 문구에서 역할을 정하므로(`accepted_roles`) V3의 새 항목도 코드 수정 없이 판정되고,
`add_version`이 새 항목의 이름 낱말을 키워드로 넣는다. 가짜 V3(12월 1일 시행 + 새 항목)로 시험 : 30/30 · 15/15,
같은 12월 질문이 V2 문서에선 V2, V3 문서에선 V3, 새 항목 질문은 V3 문서에서만 그 항목으로 잡힘.

## 넘겨주는 것 (B · C용)

```python
from rag.search import Searcher
s = Searcher("V2")                       # 한 번만 만든다 (로드 1초 안쪽)
res = s.retrieve(question, k=5)
res["version"]   # {"item", "basis", "basis_date", "applied_version": "V1"/"V2"/None, "label": "V1"/"V2"/"해당 없음"/"추가 확인"/"규정 범위 밖", "status": "확정"/"추가 확인"/"개정 항목 아님"/"규정 범위 밖", "reason", ...}
res["hits"]      # [{"page", "title", "articles", "tables", "rows"(맞은 표 행), "why", "amended", "text"}, ...]
```

- 답변의 '적용 판본' 칸에는 `label`을 쓴다 (개정 항목이 아니면 "해당 없음")
- `status == "추가 확인"`이면 생성 전에 되묻기(제8조) : `reason`에 필요한 날짜 종류가 적혀 있다
- 인용 : `hits[i]["articles"]` · `tables` · `rows` · `page`를 그대로 쓰면 '조항 · 표 행 · 쪽' 형식이 맞는다
- 문서 판본(`Searcher("V2")`)과 적용 판본(`applied_version`)은 다르다. V2 합본 안에 V1 수치가 부칙 표 54~57로 들어 있다

## A-7 : V3가 배포되면

코드는 고치지 않는다. 판본 정보는 전부 `config/versions.json`에 있고, 아래 두 스크립트가 채운다.

```bash
python -m rag.extract --doc V3 --pdf pdf/rulebook_V3.pdf                      # 1. 추출, report.txt 경고 확인
python -m scripts.add_version --doc V3 --pdf pdf/rulebook_V3.pdf              # 2. 자동 탐지 결과 확인만
python -m scripts.add_version --doc V3 --pdf pdf/rulebook_V3.pdf --write      # 3. 설정에 반영
python -m rag.index --doc V3 [--embed BAAI/bge-m3]                            # 4. 색인
python -m eval.eval_retrieval --doc V3                                        # 5. 품질
python -m eval.eval_version --doc V3                                          #    판본 규칙 (V3 시행일 이후 날짜 질문을 더 넣어 볼 것)
python -m scripts.diff_versions --old V2 --new V3 --questions eval/questions_seed.jsonl   # 6. 차이 → C
```

`add_version`이 자동으로 찾는 것과 사람이 볼 것

| 찾는 것 | 어디서 | 못 찾거나 바뀌었을 때 |
|---|---|---|
| V3 시행일 | 본문의 'V3는 YYYY년 M월 D일 00시부터' (부칙 제1조) | `--start YYYY-MM-DD` |
| 개정 항목 · 기준일 · 대조표 쪽 | 열이 '항목 · 기준일 · 대조표'인 표 (V2의 표 07) | 새 항목은 `items`에 키워드 없이 들어감 → `any`를 채운다 |
| 시행 구간 표 쪽 | 열이 '구간 · 적용'인 표 (V2의 표 53) | report.txt · PDF 확인 |
| 구조가 V2와 같은가 | 표 번호 = 쪽 번호 비율, 머리말 · 꼬리말 잔여 | 다르면 `rag/extract.py`의 정규식이나 '쪽 = 조각' 가정을 손본다 |

설정 구조 (문서별) :

```json
"versions": [{"name": "V1", "start": "2026-06-01"}, {"name": "V2", "start": "2026-10-01"}, {"name": "V3", "start": "…"}],
"docs": {
  "V2": {"versions": ["V1", "V2"],       "rule_pages": {…}, "compare_pages": {"가족행사 공동비 상한": 54, …}},
  "V3": {"versions": ["V1", "V2", "V3"], "rule_pages": {…}, "compare_pages": {…}}
}
```

같은 질문이라도 `Searcher("V2")`는 V2 문서가 아는 판본 안에서, `Searcher("V3")`는 V3까지 놓고 판정한다.
V2/V3 비교 실험은 `--doc`만 바꿔 같은 질문 세트를 돌리면 된다. 문서마다 정답이 다르면 질문 파일에 `"expected_version": {"V2": "V2", "V3": "V3"}`.

V2 데이터로 꾸민 가짜 V3(시행일 한 줄 추가 · 생신 상한 22→20만 원 · 표 07 항목 하나 교체)로 이 절차를 미리 돌려 봤다 :
시행일 · 새 항목 · 없어진 항목 · 바뀐 조항 2개 · 바뀐 표 행 1개를 모두 잡았고, 12월 날짜 질문이 V2 문서에선 V2, V3 문서에선 V3로 갈렸다.

## 폴더

```
config/versions.json    시행 구간 · 문서 경로 · 개정 항목 규칙(표 07)
rag/common.py           설정 읽기 · 토크나이저
rag/extract.py          A-1 · A-2 · A-3
rag/index.py            A-4 (조각 · 벡터)
rag/search.py           A-4 (검색 · 참조 · retrieve)
rag/version.py          A-6
eval/questions_seed.jsonl, eval/eval_retrieval.py   A-5
scripts/check_extract.py   (A) 추출 시험
scripts/ask_once.py        (A · B) 뽑은 쪽으로 질문 하나
scripts/add_version.py     A-7 새 판본 설정 자동 탐지
scripts/diff_versions.py   A-7 판본 차이 · 같은 질문 판정 차이
notebooks/role_A_colab.ipynb   코랩에서 전부
data/<doc>/             생성물 (V2는 git에 포함, 벡터 emb.npy 만 제외)
results/                평가 결과 (git 제외)
```
