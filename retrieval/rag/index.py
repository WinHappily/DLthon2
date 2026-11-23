"""A-4 : 색인 만들기. 키워드(BM25)는 검색할 때 바로 만들고, 여기서는 의미 검색용 벡터만 만든다.

    python -m rag.index --doc V2                       # 벡터 없이 (BM25만)
    python -m rag.index --doc V2 --embed BAAI/bge-m3   # RunPod · 코랩 GPU 에서

결과 (data/<doc>/) : chunks.jsonl (쪽 조각 + 표 행 조각), emb.npy · emb_meta.json (--embed 일 때)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import ROOT, load_config, read_jsonl


def build_chunks(data_dir: Path) -> list[dict]:
    pages = read_jsonl(data_dir / "pages.jsonl")
    rows = read_jsonl(data_dir / "rows.jsonl")
    chunks = [{"id": p["id"], "kind": "page", "page": p["page"], "text": p["text"]} for p in pages]
    for r in rows:
        chunks.append({"id": f"{r['doc']}-p{r['page']:02d}-{r['table'].replace(' ', '')}-r{r['row_no']}",
                       "kind": "row", "page": r["page"], "table": r["table"], "row": r["row"], "text": r["text"]})
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--embed", help="sentence-transformers 모델 이름 (예 : BAAI/bge-m3)")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config()
    data_dir = ROOT / cfg["docs"].get(args.doc, {}).get("data", f"data/{args.doc}")
    chunks = build_chunks(data_dir)
    with open(data_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"조각 {len(chunks)}개 (쪽 {sum(c['kind'] == 'page' for c in chunks)} + 표 행 {sum(c['kind'] == 'row' for c in chunks)})")

    if args.embed:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.embed)
        emb = model.encode([c["text"] for c in chunks], batch_size=args.batch,
                           normalize_embeddings=True, show_progress_bar=True)
        np.save(data_dir / "emb.npy", emb.astype("float32"))
        (data_dir / "emb_meta.json").write_text(json.dumps(
            {"model": args.embed, "dim": int(emb.shape[1]), "ids": [c["id"] for c in chunks]}, ensure_ascii=False),
            encoding="utf-8")
        print(f"벡터 {emb.shape} -> {data_dir / 'emb.npy'}")


if __name__ == "__main__":
    main()
