"""A-4 : 하이브리드 검색 + 참조 따라가기 + 판본 규칙 쪽 추가.

    python -m rag.search "어머님 생신 식사와 선물을 서로 다른 카드로 결제했어" --doc V2

retrieve() 가 B(서빙) · C(프롬프트)에 넘기는 함수다. 돌려주는 것 :
  version  : rag.version.decide() 결과 (적용 판본 · 기준일 · 상태)
  hits     : 쪽 목록. 쪽마다 점수 · 이유 · 맞은 표 행 · 조항 · 표 · 본문
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from .common import ROOT, load_config, read_jsonl, tokenize
from .version import decide

RRF_K = 60


class Searcher:
    def __init__(self, doc: str, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.doc = doc
        self.dir = ROOT / self.cfg["docs"].get(doc, {}).get("data", f"data/{doc}")
        self.pages = {p["page"]: p for p in read_jsonl(self.dir / "pages.jsonl")}
        self.amap = json.loads((self.dir / "article_map.json").read_text(encoding="utf-8"))
        cpath = self.dir / "chunks.jsonl"
        if not cpath.exists():
            raise SystemExit(f"{cpath} 가 없습니다. 먼저 python -m rag.index --doc {doc}")
        self.chunks = read_jsonl(cpath)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.index_terms = self._load_index_terms()
        self.dense = self._load_dense()

    def _load_index_terms(self) -> list[tuple[list[str], set[int]]]:
        """표 60 근거 색인 : 검색어 → 조항 · 표 쪽."""
        out = []
        for r in read_jsonl(self.dir / "rows.jsonl"):
            if r["table"] != "표 60":
                continue
            vals = list(r["cells"].values())
            terms = [t.strip() for t in re.split(r"[·,]", vals[0]) if t.strip()]
            pages = set()
            for cell in vals[1:]:
                for a, b in re.findall(r"제(\d+)조(?:\s*~\s*제(\d+)조)?", cell):
                    for n in range(int(a), int(b or a) + 1):
                        if f"제{n}조" in self.amap:
                            pages.add(self.amap[f"제{n}조"])
                for m in re.finditer(r"표\s*([\d·~\s]+)", cell):
                    nums, prev = [], None
                    for tok in re.findall(r"\d+|~", m.group(1)):
                        if tok == "~":
                            prev = "range"
                        elif prev == "range" and nums:
                            nums.extend(range(nums[-1] + 1, int(tok) + 1))
                            prev = None
                        else:
                            nums.append(int(tok))
                    pages.update(n for n in nums if 1 <= n <= len(self.pages))
            out.append((terms, pages))
        return out

    def _load_dense(self):
        meta = self.dir / "emb_meta.json"
        if not meta.exists():
            return None
        import numpy as np
        from sentence_transformers import SentenceTransformer
        m = json.loads(meta.read_text(encoding="utf-8"))
        return {"model": SentenceTransformer(m["model"]), "emb": np.load(self.dir / "emb.npy")}

    def _ranks(self, query: str) -> dict[int, float]:
        rrf: dict[int, float] = {}
        bm = self.bm25.get_scores(tokenize(query))
        for r, i in enumerate(sorted(range(len(bm)), key=lambda i: -bm[i])[:50]):
            if bm[i] > 0:
                rrf[i] = rrf.get(i, 0) + 1 / (RRF_K + r + 1)
        if self.dense:
            q = self.dense["model"].encode([query], normalize_embeddings=True)[0]
            sims = self.dense["emb"] @ q
            for r, i in enumerate(sims.argsort()[::-1][:50]):
                rrf[i] = rrf.get(i, 0) + 1 / (RRF_K + r + 1)
        return rrf

    def search(self, query: str, k: int = 5, follow_refs: bool = True, max_refs: int = 4,
               use_index_terms: bool = True) -> list[dict]:
        rrf = self._ranks(query)
        page_score: dict[int, float] = {}
        rows: dict[int, list[str]] = {}
        why: dict[int, list[str]] = {}
        for i, s in sorted(rrf.items(), key=lambda x: -x[1]):
            c = self.chunks[i]
            page_score[c["page"]] = max(page_score.get(c["page"], 0), s)
            if c["kind"] == "row" and len(rows.setdefault(c["page"], [])) < 3:
                rows[c["page"]].append(f"{c['table']} · {c['row']}")
        if use_index_terms:
            bonus = 1 / (RRF_K + 1)
            for terms, pages in self.index_terms:
                hit = [t for t in terms if t in query]
                for p in pages if hit else []:
                    page_score[p] = page_score.get(p, 0) + bonus * 0.5
                    why.setdefault(p, []).append(f"색인어 '{hit[0]}'(표 60)")
        primary = sorted(page_score, key=lambda p: -page_score[p])[:k]
        hits = [self._hit(p, page_score[p], ["검색 %d위" % (n + 1)] + why.get(p, []), rows.get(p, []))
                for n, p in enumerate(primary)]
        if follow_refs:
            counts: dict[int, list[str]] = {}
            for p in primary:
                for ref in self.pages[p]["refs"]["articles"] + self.pages[p]["refs"]["tables"]:
                    tp = self.amap.get(ref)
                    if tp and tp not in primary:
                        counts.setdefault(tp, []).append(f"p{p}의 {ref}")
            for tp, src in sorted(counts.items(), key=lambda x: -len(x[1]))[:max_refs]:
                hits.append(self._hit(tp, 0.0, ["참조 : " + ", ".join(src[:3])], []))
        return hits

    def _hit(self, page: int, score: float, why: list[str], rows: list[str]) -> dict:
        p = self.pages[page]
        return {"page": page, "score": round(score, 5), "why": why, "rows": rows, "title": p["title"],
                "articles": p["articles"], "tables": p["tables"], "amended": p["amended"], "text": p["text"]}

    def retrieve(self, question: str, k: int = 5, follow_refs: bool = True, max_refs: int = 4) -> dict:
        ver = decide(question, self.cfg, self.doc)
        hits = self.search(question, k=k, follow_refs=follow_refs, max_refs=max_refs)
        have = {h["page"] for h in hits}
        for p in ver["pages"]:
            if p not in have:
                hits.append(self._hit(p, 0.0, [f"판본 규칙({ver['item']})"], []))
                have.add(p)
        return {"question": question, "doc": self.doc, "version": ver, "hits": hits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--doc", default="V2")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--no-refs", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = Searcher(a.doc).retrieve(a.question, k=a.k, follow_refs=not a.no_refs)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return
    v = res["version"]
    print(f"[판본] {v['status']} · 항목 {v['item']} · 기준 {v['basis']} · 날짜 {v['basis_date']} → {v['applied_version']}")
    print(f"       {v['reason']}")
    for h in res["hits"]:
        print(f"p{h['page']:02d} {h['title']:<18} {' · '.join(h['articles'] + h['tables'])}  [{'; '.join(h['why'])}]"
              + (f"  행: {', '.join(h['rows'])}" if h["rows"] else ""))


if __name__ == "__main__":
    main()
