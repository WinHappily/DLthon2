"""검색용 질문(clean_query 결과)의 벡터를 미리 만들어 data/<doc>/qemb.npz 로 저장한다 (코랩 GPU에서).

    python -m scripts.dump_query_emb --questions <개발문항_150.jsonl> <잡음문항_90.jsonl> ...

로컬(torch를 못 쓰는 PC)에서도 emb.npy + qemb.npz 만 있으면 BM25 + bge-m3 검색을 그대로 재현해
검색 개선을 시험할 수 있다. 공개 100 파일은 넣지 않는다(동결 전 검색 결과를 보지 않는다는 팀 결정).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rag.query import clean_query
from rag.search import Searcher
from scripts.export_retrieval import load_questions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", nargs="+", required=True)
    ap.add_argument("--doc", default="V2")
    a = ap.parse_args()
    s = Searcher(a.doc)
    if not s.dense or not hasattr(s.dense["model"], "tokenizer"):
        raise SystemExit("벡터 모델이 없습니다 : 2절 색인을 --embed BAAI/bge-m3 로 만든 뒤 다시")
    texts = []
    for p in a.questions:
        if Path(p).exists():
            texts += [clean_query(q["question"], s.dir) for q in load_questions(Path(p))]
        else:
            print("없음(건너뜀) :", p)
    texts = list(dict.fromkeys(texts))
    vecs = s.dense["model"].encode(texts, normalize_embeddings=True, batch_size=16).astype("float32")
    out = s.dir / "qemb.npz"
    np.savez(out, texts=np.array(texts), vecs=vecs)
    print(f"질문 벡터 {vecs.shape} -> {out}")


if __name__ == "__main__":
    main()
