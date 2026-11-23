"""질문 정리 : 전각 숫자 · 이모지 · 'ㅋㅋ' 정리, 긴 잡담 글에서 사건 문장만 남기기.

    python -m rag.query "<질문>"            # 정리된 검색용 질문 보기

공개 평가셋에는 질문 하나에 잡담 문단이 수십 개 섞인 글(최대 약 8천 자)이 있다.
그대로 BM25 · 임베딩에 넣으면 잡담 낱말이 점수를 흐린다.
- 모든 질문 : NFKC 정규화(２０２６ → 2026), 이모지 · ㅋㅎㅠ 반복 제거
- LONG 자 넘고 3문장 이상인 질문 : 문장마다 점수(숫자 · 날짜 / 규약 용어가 있는 질문 문장 / 용어 2개 + 표 60 색인어가 있는 서술)로 사건 문장만 남긴다 (11.19 C 3-3, 잡음 세트 c로만 확인).
  규약 용어는 규약집의 장 · 조항 · 표 제목과 표 행 이름에서 자동으로 뽑는다(문서마다, V3도 그대로).
모델에 넣는 원문은 바꾸지 않는다. 검색 · 판본 판정에만 쓴다.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

LONG = 150
MIN_HITS = 1
EMOJI = re.compile("[🀀-🫿☀-➿️]+")
LAUGH = re.compile("[ㅋㅎㅠㅜㄱᄏ휴ᅮᄀ]{2,}|[?!]{3,}")   # ㅋㅎㅠㅜㄱ, NFKC 뒤 자모 포함
WORD = re.compile(r"[가-힣]{2,}")
GLUED = re.compile(r"[가-힣]{10,}")   # 띄어쓰기 없이 붙여 쓴 구절 (잡음 변형 a : 표기 · 띄어쓰기)
SENT = re.compile(r"(?<=[.?!。])\s+|\n+")
# 규약 제목에도 나오지만 잡담에서 흔한 말 : 용어로 치지 않는다
GENERIC = {"확인", "기록", "기준", "내용", "경우", "사용", "관리", "운영", "항목", "조건", "방식", "처리", "문서", "판단",
           "적용", "사건", "날짜", "시간", "표현", "정리", "문의", "질문", "예시",
           # 일상 기능어 : 규약 제목에 있어도 잡담에서 흔하다
           "같은", "같이", "다른", "하는", "사람", "사람이", "쓰는", "다음", "모두", "지금", "순서", "용도", "있나",
           "각각", "내부", "여부", "원을", "원과", "처음", "이상", "전에", "이후", "부분", "자체", "상태", "방법",
           "이번", "오늘", "어제", "내일", "주말"}
POLITE_END = re.compile(r"(니다|세요|어요|아요)[.!]?$")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = EMOJI.sub(" ", t)
    t = LAUGH.sub(" ", t)
    return re.sub(r"[ \t]+", " ", t).strip()


@lru_cache(maxsize=8)
def lexicon(data_dir: str) -> frozenset[str]:
    import json
    d = Path(data_dir)
    words: set[str] = set()
    for line in (d / "pages.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        text = json.loads(line)["text"]
        for h in re.findall(r"^#+ (.+)$", text, re.M):          # 장 · 조항 · 표 제목
            words.update(WORD.findall(h))
    for line in (d / "rows.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            words.update(WORD.findall(r.get("row", "")))
            words.update(WORD.findall(r.get("table_title", "")))
    return frozenset(w for w in words if w not in GENERIC)


@lru_cache(maxsize=8)
def index_terms(data_dir: str) -> frozenset[str]:
    """표 60 근거 색인의 검색어 (규약 주제를 대표하는 말 : 생신 · 숙박 · 여행 · 방문 · 취소 …)."""
    import json
    out: set[str] = set()
    for line in (Path(data_dir) / "rows.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("table") == "표 60":
                first = list(r["cells"].values())[0]
                out.update(t.strip() for t in re.split(r"[·,]", first) if t.strip())
    return frozenset(out)


@lru_cache(maxsize=16)
def _pattern(words: frozenset[str]) -> re.Pattern:
    alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(r"(?<![가-힣])(?:" + alt + ")")


def _hits(sentence: str, words: frozenset[str]) -> int:
    """낱말 첫머리에서 맞은 서로 다른 용어 수 ('화분에' 속 '분에', '충전기' 속 '전기'는 세지 않는다).
    띄어쓰기 없이 10자 넘게 붙여 쓴 구절('규약의내부대출상한안에들어와')은 그 안에서도 찾는다 (V3 잡음 N018)."""
    if not words:
        return 0
    found = set(_pattern(words).findall(sentence))
    for tok in GLUED.findall(sentence):
        found |= {w for w in words if w in tok}
    return len(found)


def _topic(sentence: str, idx: frozenset[str]) -> int:
    """표 60 색인어(규약 주제어)는 띄어쓰기 없이 붙여 써도 찾는다 ('시어머니생신모임')."""
    return sum(1 for w in idx if w in sentence)


NUMBER = re.compile(r"\d")
QUESTION = re.compile(r"\?|(?:까|니|냐|나|는지|건가|거임|되나)\s*[.!~]*\s*$")   # 반말 서술(-돼 · -해 · -야)은 질문으로 보지 않는다
PARA = re.compile(r"\n\s*\n")


def _is_event(s: str, lex: frozenset[str], idx: frozenset[str]) -> bool:
    """사건 문장 : 숫자 · 날짜 · 금액이 있거나, 규약 용어가 든 질문, 또는 규약 주제어가 든 서술."""
    if NUMBER.search(s):
        return True
    h = _hits(s, lex) + _topic(s, idx)
    if QUESTION.search(s):
        return h >= 1
    return (_topic(s, idx) >= 1 or h >= 3) and not POLITE_END.search(s)


def _off_topic_question(s: str, lex: frozenset[str], idx: frozenset[str]) -> bool:
    """'근데 요즘 볼 만한 영화 뭐 있어?' : 규약 용어도 숫자도 없는 질문 (두 번째 질문)."""
    return bool(QUESTION.search(s)) and not NUMBER.search(s) and _hits(s, lex) + _topic(s, idx) == 0


def clean_query(text: str, data_dir: str | Path | None = None) -> str:
    """검색 · 판본 판정용 질문. 짧은 질문은 정규화만 한다 (모델에는 원문을 준다, 팀 결정 4).
    긴 글 : 문단(빈 줄로 나뉜 덩어리)마다 사건 문장이 하나라도 있으면 그 문단을 남기고(두 번째 질문은 뺌),
    사건 문장이 없는 문단은 버린다. 문단 구분이 없으면 문장 단위로 같은 기준을 쓴다."""
    t = normalize(text)
    sents = [s.strip() for s in SENT.split(t) if s.strip()]
    if data_dir is None:   # 폴더를 안 주면 RAG_DOC(없으면 V2) (C의 eval_version.py처럼 clean_query(질문)만 불러도 검색과 같은 결과)
        import os
        from .common import ROOT
        data_dir = ROOT / "data" / os.environ.get("RAG_DOC", "V2")
        if not (data_dir / "pages.jsonl").exists():
            data_dir = None
    if data_dir is None or len(t) <= LONG or len(sents) < 3:
        return t
    lex, idx = lexicon(str(data_dir)), index_terms(str(data_dir))
    keep: list[str] = []
    paras = [p for p in PARA.split(t) if p.strip()]
    for p in paras if len(paras) > 1 else [t]:
        ss = [s.strip() for s in SENT.split(p) if s.strip()]
        ev = [i for i, s in enumerate(ss) if _is_event(s, lex, idx)]
        if len(paras) > 1 and ev:   # 문단 안에서는 첫 사건 문장 ~ 마지막 사건 문장까지 (앞뒤 잡담 · 끝에 붙은 질문 제외)
            keep += [s for s in ss[ev[0]:ev[-1] + 1] if not _off_topic_question(s, lex, idx)]
        elif len(paras) <= 1:
            keep += [s for s in ss if _is_event(s, lex, idx)]
    keep = list(dict.fromkeys(keep))
    return "\n".join(keep) if keep else t


if __name__ == "__main__":
    from .common import ROOT
    q = " ".join(sys.argv[1:]) or sys.stdin.read()
    out = clean_query(q, ROOT / "data" / "V2")
    print(f"[{len(q)}자 → {len(out)}자]\n{out}")
