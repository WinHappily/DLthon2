"""조항 단위 오라클(v2.1)을 A의 쪽 단위로 바꾼다 (O1p).
- 근거 단위 → 그 단위가 있는 쪽. 본문·출처 표기는 make_t2 출력과 같게(A의 pages.jsonl 그대로)
- 고정 쪽(A 설정 fixed_pages)은 T2처럼 맨 뒤. 58·59쪽은 뺀다(정답이 드러나는 표)
실행: python make_oracle_page.py <v2.1 오라클.jsonl> <A 저장소 폴더> <출력.jsonl>
"""
import json, re, sys

src, repo, out = sys.argv[1:4]
P = {p["page"]: p for p in map(json.loads, open(f"{repo}/data/V2/pages.jsonl", encoding="utf-8"))}
FIXED = json.load(open(f"{repo}/config/versions.json", encoding="utf-8"))["docs"]["V2"]["fixed_pages"]
EXCLUDE = {58, 59}


def two_digit(name):  # make_t2와 같은 표기: '표 7' → '표 07'
    return re.sub(r"표\s*(\d+)", lambda m: f"표 {int(m.group(1)):02d}", name)


def key(u):  # '표 07' / '제23조' / '부칙 제1조' → 비교용
    return two_digit(u).replace(" ", "")


page_of = {}
for pg, p in P.items():
    for u in [str(a) for a in p["articles"]] + [str(t) for t in p["tables"]]:
        page_of[key(u)] = pg

RANK = {"오라클": 0, "근접": 1, "보강": 2}
with open(out, "w", encoding="utf-8") as f:
    for row in map(json.loads, open(src, encoding="utf-8")):
        why = {}
        for c in row["조각"]:
            pg = page_of[key(c["출처"])]
            if pg in FIXED or pg in EXCLUDE:
                continue
            w = c.get("경로") or "오라클"
            why[pg] = min(why.get(pg, w), w, key=RANK.get)
        pages = [(pg, why[pg]) for pg in sorted(why)] + [(pg, "고정 조각(공통 근거)") for pg in FIXED]
        chunks = [{"출처": two_digit("·".join([str(x) for x in P[pg]["articles"]] + [str(x) for x in P[pg]["tables"]])),
                   "쪽": pg, "본문": P[pg]["text"], "경로": w} for pg, w in pages]
        f.write(json.dumps({"문항번호": row["문항번호"], "조각": chunks}, ensure_ascii=False) + "\n")
print("저장:", out)
