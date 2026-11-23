"""(A · B) 추출한 쪽을 그대로 프롬프트에 붙여 질문 하나에 답 받아 보기.

vLLM 같은 OpenAI 호환 서버에 보낸다. 서버는 따로 띄운다(README 'RunPod에서 시험' 절).

    # 쪽을 검색으로 고르기 (기본)
    python -m scripts.ask_once --model Qwen/Qwen2.5-32B-Instruct-AWQ --base-url http://localhost:8000/v1
    # 쪽을 손으로 지정
    python -m scripts.ask_once --pages 41 42 54 14 13 --model ...
    # 서버 없이 프롬프트만 보기
    python -m scripts.ask_once --dry-run

기본 질문은 표 59의 2행(정답이 규약집에 있음) :
  V2, 생신 선물 15만 원 + 식사 7만 원을 다른 카드로 결제 예정 → 같은 행사 22만 원으로 합산,
  상한과 같아 공동비 대상, 20만 원 이상이라 사전 합의 필요 (제3조 · 제23조 · 제79조 · 표 41)
결과 : 화면 + results/ask_once.jsonl 에 한 줄씩 쌓임 (모델 · 쪽 · 시간 · 토큰 · 답)
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

from rag.brief import brief_text, build_brief
from rag.common import ROOT
from rag.prompt import build_messages
from rag.search import Searcher

DEFAULT_Q = "10월 15일에 시어머니 생신 선물 15만 원과 식사 7만 원을 서로 다른 카드로 결제할 예정이야. 월 잔액은 충분해. 공동비로 처리할 수 있어?"
DEFAULT_EXPECT = "사전 합의 필요 · V2 · 같은 행사 22만 원 합산, 상한(22만 원)과 같아 공동비 대상, 20만 원 이상이라 첫 결제 전 합의 · 제3조 · 제23조 · 제79조 · 표 41"

SYSTEM = """너는 신혼부부 나미와 감석의 '공동생활 규약집'으로 판정하는 상담원이다.
아래 [규약 발췌]에 적힌 내용만 근거로 쓴다. 발췌에 없는 수치나 조항을 지어내지 않는다.
판단 순서(표 05) : 1 기준일과 판본 → 2 개별 조건과 예외 → 3 합산액과 상한 → 4 동의와 예산 → 5 증빙과 기록.
금액 단위는 만 원, '이상'은 기준 포함, '초과'는 기준보다 큰 경우(제3조).

답은 반드시 아래 다섯 줄 형식으로만 쓴다.
판정 : 가능 / 사전 합의 필요 / 공동 처리 불가 / 추가 확인 / 규정에 없음 중 하나
적용 판본 : V1 또는 V2 (기준일 종류와 날짜), 개정 항목이 아니면 '해당 없음'
근거 : 조항 번호 · 표 번호와 행 이름 · 쪽 번호
다음 행동 : 한 문장
되묻기 : 판정을 가르는 정보가 빠졌으면 무엇을 묻는지, 없으면 '없음'"""


def build_prompt(question: str, hits: list[dict], version: dict | None) -> list[dict]:
    parts = []
    for h in hits:
        parts.append(f"<{h['page']}쪽>\n{h['text']}")
    ver = ""
    if version:
        ver = (f"\n[판본 규칙 결과] 상태 {version['status']} · 항목 {version['item']} · 기준일 종류 {version['basis']}"
               f" · 기준일 {version['basis_date']} · 적용 판본 {version.get('label') or version['applied_version']}\n"
               f"  이유 : {version['reason']}\n")
    user = f"[규약 발췌]\n" + "\n\n".join(parts) + f"\n{ver}\n[질문]\n{question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def chat(base_url: str, model: str, messages: list[dict], max_tokens: int, timeout: int) -> dict:
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    data["_sec"] = time.perf_counter() - t0
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default=DEFAULT_Q)
    ap.add_argument("--expect", default=None, help="정답 메모 (기본 질문이면 자동)")
    ap.add_argument("--doc", default="V2")
    ap.add_argument("--pages", type=int, nargs="*", help="붙일 쪽을 직접 지정. 없으면 retrieve() 로 고름")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--max-refs", type=int, default=2)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true", help="서버에 보내지 않고 프롬프트만 출력")
    ap.add_argument("--no-brief", action="store_true", help="판정 메모 빼기 (비교용)")
    ap.add_argument("--old-prompt", action="store_true", help="11.17 다섯 줄 프롬프트 (비교용)")
    a = ap.parse_args()

    s = Searcher(a.doc)
    res = s.retrieve(a.question, k=a.k, max_refs=a.max_refs)
    if a.pages:
        res["hits"] = [s._hit(p, 0.0, ["직접 지정"], []) for p in a.pages]
    hits, version = res["hits"], res["version"]
    if a.old_prompt:      # 11.17 프롬프트 (판정 메모 · 이유 줄 없음) : 비교용
        msgs = build_prompt(a.question, hits, version)
    else:
        bt = None if a.no_brief else brief_text(build_brief(s, res))
        msgs = build_messages(a.question, res, bt)
    chars = sum(len(m["content"]) for m in msgs)
    expect = a.expect or (DEFAULT_EXPECT if a.question == DEFAULT_Q else "")

    print(f"[질문] {a.question}")
    print(f"[붙인 쪽] {[h['page'] for h in hits]}  (프롬프트 {chars:,}자)")
    print(f"[판본 규칙] {version['status']} · {version.get('label') or version['applied_version']} · {version['reason']}")
    if a.dry_run:
        print("\n" + msgs[-1]["content"][-2500:])   # 마지막 사용자 메시지 끝부분 (판정 메모 + 질문)
        return

    data = chat(a.base_url, a.model, msgs, a.max_tokens, a.timeout)
    ans = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    print(f"\n[모델] {a.model}  {data['_sec']:.1f}초  입력 {usage.get('prompt_tokens')} · 출력 {usage.get('completion_tokens')} 토큰")
    print(f"\n{ans}")
    if expect:
        print(f"\n[정답 메모] {expect}")

    out = ROOT / "results" / "ask_once.jsonl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": a.model, "doc": a.doc,
                            "question": a.question, "pages": [h["page"] for h in hits],
                            "version": version, "sec": round(data["_sec"], 2), "usage": usage,
                            "prompt": "old" if a.old_prompt else ("no-brief" if a.no_brief else "brief"),
                            "answer": ans, "expect": expect}, ensure_ascii=False) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
