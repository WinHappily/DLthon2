"""A-4 : 하이브리드 검색 + 참조 따라가기 + 판본 규칙 쪽 추가.

    python -m rag.search "어머님 생신 식사와 선물을 서로 다른 카드로 결제했어" --doc V2

retrieve() 가 B(서빙 · 프롬프트)에 넘기는 함수다. 돌려주는 것 :
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
from .query import clean_query
from .version import decide

RRF_K = 60
HUB_REFS = False      # True : 목차형 쪽(표 60 · 공개 사례)에서도 참조를 따라간다 (11.19 이전 동작, 비교용)


class _CachedEncoder:
    """미리 만든 질문 벡터 꾸러미. 없는 질문이면 KeyError (검색용 질문이 바뀌면 코랩에서 다시 만든다)."""

    def __init__(self, path: Path):
        import numpy as np
        z = np.load(path, allow_pickle=False)
        self.vec = dict(zip(z["texts"].tolist(), z["vecs"]))

    def encode(self, texts, normalize_embeddings=True):
        import numpy as np
        return np.stack([self.vec[t] for t in texts])


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
        # 목차형 쪽 : 근거 색인(표 60) · 공개 사례(case_pages). 참조가 13~52개라 참조 칸을 다 차지한다 → 참조를 따라가지 않는다
        self.hub_refs = HUB_REFS
        self.hub_pages = set(self.cfg["docs"].get(doc, {}).get("case_pages", [])) | {
            c["page"] for c in self.chunks if c.get("table") == "표 60"}

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
        if not meta.exists() or not (self.dir / "emb.npy").exists():
            return None
        import numpy as np
        m = json.loads(meta.read_text(encoding="utf-8"))
        emb = np.load(self.dir / "emb.npy")
        try:
            from sentence_transformers import SentenceTransformer
            return {"model": SentenceTransformer(m["model"]), "emb": emb}
        except (ImportError, OSError):
            # 모델을 못 쓰는 곳(로컬 CPU) : 코랩에서 미리 만든 질문 벡터(qemb.npz)로 같은 계산 (scripts.dump_query_emb)
            q = self.dir / "qemb.npz"
            return {"model": _CachedEncoder(q), "emb": emb} if q.exists() else None

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
               use_index_terms: bool = True, exclude: set[int] | frozenset = frozenset()) -> list[dict]:
        rrf = {i: v for i, v in self._ranks(query).items() if self.chunks[i]["page"] not in exclude}
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
                for p in [p for p in pages if p not in exclude] if hit else []:
                    page_score[p] = page_score.get(p, 0) + bonus * 0.5
                    why.setdefault(p, []).append(f"색인어 '{hit[0]}'(표 60)")
        primary = sorted(page_score, key=lambda p: -page_score[p])[:k]
        hits = [self._hit(p, page_score[p], ["검색 %d위" % (n + 1)] + why.get(p, []), rows.get(p, []))
                for n, p in enumerate(primary)]
        if follow_refs:
            counts: dict[int, list[str]] = {}
            for p in primary:
                if p in self.hub_pages and not self.hub_refs:
                    continue
                for ref in self.pages[p]["refs"]["articles"] + self.pages[p]["refs"]["tables"]:
                    tp = self.amap.get(ref)
                    if tp and tp not in primary and tp not in exclude:
                        counts.setdefault(tp, []).append(f"p{p}의 {ref}")
            for tp, src in sorted(counts.items(), key=lambda x: -len(x[1]))[:max_refs]:
                hits.append(self._hit(tp, 0.0, ["참조 : " + ", ".join(src[:3])], []))
        return hits

    def _hit(self, page: int, score: float, why: list[str], rows: list[str]) -> dict:
        p = self.pages[page]
        return {"page": page, "score": round(score, 5), "why": why, "rows": rows, "title": p["title"],
                "articles": p["articles"], "tables": p["tables"], "amended": p["amended"], "text": p["text"]}

    def retrieve(self, question: str, k: int = 5, follow_refs: bool = True, max_refs: int = 4,
                 fixed: bool = True, cases: bool = True) -> dict:
        """cases=False : 공개 사례 쪽(설정 case_pages, 표 58 · 59)을 검색 · 참조에서 뺀다."""
        q = clean_query(question, self.dir)      # 전각 숫자 · 이모지 정리, 긴 잡담 글은 사건 문장만
        ver = decide(q, self.cfg, self.doc)
        exclude = frozenset() if cases else frozenset(self.cfg["docs"].get(self.doc, {}).get("case_pages", []))
        hits = self.search(q, k=k, follow_refs=follow_refs, max_refs=max_refs, exclude=exclude)
        have = {h["page"] for h in hits}
        extra = [(p, f"판본 규칙({ver['item']})") for p in ver["pages"]]
        topics = self.cfg["docs"].get(self.doc, {}).get("topic_pages", [])
        for t in topics:   # 총칙 쪽 : 질문 말투가 그 규정을 가리킬 때 (판본 '추가 확인'이면 상태 판정 쪽도)
            if re.search(t["pattern"], q) or (ver["status"] == "추가 확인" and "상태" in t["why"]):
                extra.append((t["page"], t["why"]))
        if fixed:          # 공통 근거 쪽 : 모든 문항에 (오라클 조건과 맞춤)
            extra += [(p, "고정 조각(공통 근거)") for p in self.cfg["docs"].get(self.doc, {}).get("fixed_pages", [])]
        for p, why in extra:
            if p not in have:
                hits.append(self._hit(p, 0.0, [why], []))
                have.add(p)
        return {"question": question, "query": q, "doc": self.doc, "version": ver, "hits": hits}


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
