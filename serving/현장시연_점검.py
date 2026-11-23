#!/usr/bin/env python3
"""현장시연.py의 실시간 검색이 평가 때 쓴 검색 결과 파일과 같은 입력을 만드는지 개발 150문항으로 대조한다 (모델 서버 불필요).
실행: python 현장시연_점검.py   (먼저 python 현장시연.py --build-index)
"""
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("demo", os.path.join(HERE, "현장시연.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
from rag.brief import brief_text, build_brief

R = m.R


def load(p):
    return {json.loads(l)["문항번호"]: json.loads(l) for l in open(os.path.join(HERE, p), encoding="utf-8") if l.strip()}


qs = load("data/개발문항_150.jsonl")
enc = m.Encoder()
for doc, rf, vf in [("V2", "data/검색결과_T2_150.jsonl", "data/판본_A규칙_150.jsonl"),
                    ("V3", "data/검색결과_T2v3_150.jsonl", "data/판본_A규칙v3_150.jsonl")]:
    st, sv, s = load(rf), load(vf), m.load_searcher(doc, enc)
    pages = msgs = vers = 0
    bad = []
    for i, q in qs.items():
        res = s.retrieve(q["질문"], k=5, max_refs=4)
        ch, v = m.to_chunks(res["hits"]), m.to_version(res["version"])
        pages += [c["쪽"] for c in ch] == [c["쪽"] for c in st[i]["조각"]]
        vers += v["판본"] == sv[i]["판본"]
        same = R.format_user(q["질문"], ch, brief_text(build_brief(s, res)), v) == R.format_user(q["질문"], st[i]["조각"], st[i]["메모"], sv[i])
        msgs += same
        if not same:
            bad.append(i)
    print(f"[{doc}] 150문항 중 검색 쪽 같음 {pages} · 판본 같음 {vers} · 모델 입력 전체 같음 {msgs}  다른 문항 {bad[:12]}")
