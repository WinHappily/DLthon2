"""입력 프롬프트 토큰 수만 잰다 (C 지시 3-4 : 가장 긴 공개 5문항이 --max-model-len 16384 안에 드는지).

    python -m scripts.count_tokens --questions <01_공개_배포평가셋_100문항.jsonl> \
        --ids PUB-086 PUB-093 PUB-091 PUB-075 PUB-028 --models Qwen/Qwen3-4B-AWQ Qwen/Qwen3-32B-AWQ

- 입력 = export_retrieval 과 같은 방식 (시스템 지시문 + 고정 조각 포함 hits + 판정 메모 + 질문 원문), chat template 적용
- 출력은 숫자만 (문항 내용 · 검색 결과는 찍지 않는다. 공개 100은 동결 전 결과를 보지 않는다는 팀 결정)
- 시스템 지시문은 A의 기본값(rag/prompt.py). B의 run_all 지시문은 길이가 다르므로 그 차이만큼 더하거나 뺀다
"""
from __future__ import annotations

import argparse

from rag.brief import brief_text, build_brief
from rag.prompt import build_messages
from rag.search import Searcher
from scripts.export_retrieval import load_questions
from pathlib import Path

LIMIT = 16384 - 1536   # 서버 한도 - 최대 출력


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3-4B-AWQ", "Qwen/Qwen3-32B-AWQ"])
    ap.add_argument("--doc", default="V2")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    toks = {m: AutoTokenizer.from_pretrained(m, trust_remote_code=True) for m in a.models}
    qs = {q["id"]: q["question"] for q in load_questions(Path(a.questions))}
    s = Searcher(a.doc)
    over = []
    for i in a.ids:
        res = s.retrieve(qs[i])
        msgs = build_messages(qs[i], res, brief_text(build_brief(s, res)))
        parts = []
        for m, t in toks.items():
            # 글자로 만든 뒤 토큰화 (transformers 5는 tokenize=True 일 때 dict를 돌려줘 len이 2가 된다)
            text = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            n = len(t(text, add_special_tokens=False)["input_ids"])
            parts.append(f"{m.split('/')[-1]} {n:,}")
            if n > LIMIT:
                over.append(f"{i} {m}")
        print(f"{i}: " + " / ".join(parts) + f"  (입력 {sum(len(x['content']) for x in msgs):,}자)")
    print(f"한도 {LIMIT:,} 토큰(16,384 - 출력 1,536) 넘는 것 : {over or '없음'}")


if __name__ == "__main__":
    main()
