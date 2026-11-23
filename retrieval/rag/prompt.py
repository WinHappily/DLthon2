"""답변 프롬프트 (A가 제안하는 기본값, B가 바꿔 써도 된다).

    from rag.prompt import build_messages
    msgs = build_messages(question, res, brief)      # res = Searcher.retrieve(), brief = build_brief()

정답 예시처럼 '결론 + 이유(수치 비교 · 오해 반박) + 근거'가 나오도록 :
  1) 판정 메모(rag.brief)로 금액 합계 · 그 판본의 수치 · 핵심 문장을 먼저 준다
  2) 이유 줄을 '계산/수치 비교 → 질문 속 다른 조건이 결론을 바꾸는지'로 강제한다
예시 답과 실제 문항의 숫자는 넣지 않는다 (11.19 정리) :
  - 예시 답이 개발 문항 D015와 글자까지 같아 정답이 새어 나갔다 (B 지적)
  - 지시문의 '72만 원 · 70만 원' 예는 운영진 공개 예시의 풀이라 뺐다 (공개 100문항은 봉인)
모델에 넣는 질문은 원문 그대로다.
"""
from __future__ import annotations

SYSTEM = """너는 신혼부부 나미와 감석의 '공동생활 규약집'으로 판정하는 상담원이다.
[규약 발췌]에 적힌 내용만 근거로 쓴다. 발췌에 없는 수치나 조항을 지어내지 않는다.
[판정 메모]는 검색기가 뽑은 사실이다. 발췌와 맞는지 확인해서 쓴다.
판단 순서(표 05) : 1 기준일과 판본 → 2 개별 조건과 예외 → 3 합산액과 상한 → 4 동의와 예산 → 5 증빙과 기록.
금액 단위는 만 원, '이상'은 기준 포함, '초과'는 기준보다 큰 경우(제3조).

답은 아래 형식으로 쓴다.
판정 : 질문이 답 형식(예/아니오, 선택지, 숫자, 상태만, 목록)을 정했으면 그 형식으로 먼저 답하고, 판정 질문이면 가능 / 사전 합의 필요 / 공동 처리 불가 / 추가 확인 / 규정에 없음 중 하나를 붙인다
이유 : 2~3문장. 첫 문장은 결론을 가른 계산이나 수치 비교(합산액과 상한 · 기준 금액을 숫자로 비교). 다음 문장은 질문에 나온 다른 조건(카드 분리, 사전 합의, 예산, 할인, 날짜 등)이 결론을 바꾸는지 짚는다
적용 판본 : V1 또는 V2 (기준일 종류와 날짜), 개정 항목이 아니면 '해당 없음'
근거 : 쪽 번호, 조항 · 표 (예: ○○쪽 제○조 · 표 ○○)
다음 행동 : 한 문장. 판정을 가르는 정보가 빠졌으면 무엇을 묻는지 적는다"""


def build_messages(question: str, res: dict, brief_txt: str | None = None,
                   max_pages: int | None = None) -> list[dict]:
    hits = res["hits"][:max_pages] if max_pages else res["hits"]
    parts = [f"<{h['page']}쪽>\n{h['text']}" for h in hits]
    user = "[규약 발췌]\n" + "\n\n".join(parts)
    if brief_txt:
        user += "\n\n" + brief_txt
    user += f"\n\n[질문]\n{question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
