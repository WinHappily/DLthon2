# 결과 파일 전부를 간단히 채점해 results/평가_요약.md 로 쓴다 (CPU만, 몇 초). 정식 채점은 score_cell.py
# 주지표 = 상태와 적용_판본이 둘 다 정답. R3 = '추가 확인'인데 누락이 합의·동의뿐이고 아직 안 한 일이면 '사전 합의 필요'
# V3 실험: 정답지는 V2 기준이라, A 규칙의 V3 판본이 V2 정답과 다른 15문항은 따로 뺀 점수도 적는다(V3 정답 나오면 다시 채점)
import argparse, collections, csv, glob, json, os, re

W = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--gold", nargs="+", default=[f"{W}/data/개발정답_150_N90_F30.csv"], help="정답 CSV들 (V3 정답이 오면 그 파일들)")
ap.add_argument("--exp", default="", help="실험ID에 이 글자가 든 것만 (예: V3)")
ap.add_argument("--out", default=f"{W}/results/평가_요약.md")
args = ap.parse_args()
G = {}
for g in args.gold:
    G.update({r["문항번호"]: r for r in csv.DictReader(open(g, encoding="utf-8-sig"))})
V3LBL = {}
for f in glob.glob(f"{W}/data/판본_A규칙v3_*.jsonl"):
    if "train800" not in f:
        V3LBL.update({json.loads(l)["문항번호"]: json.loads(l)["판본"] for l in open(f, encoding="utf-8")})
CHANGED = {q for q, v in V3LBL.items() if q in G and v != G[q]["적용_판본"]}
PAST = re.compile(r"(결제했|결제를 했|샀어|샀는데|구매했|냈어|냈는데|보냈어|지불했|이미 [^ ]*(했|샀|냈|보냈|결제)|먹었|다녀왔|했어\. ?(이제|이걸)|처리해도|정산해도|공동비로 (처리|넣)|공동 ?돈으로 (처리|정산))")


def parse(r):
    try:
        return json.loads(r["출력_원문"]) if r.get("출력_원문") else None
    except Exception:
        return None


def r3(q, o):
    st = o.get("상태")
    xs = [str(x) for x in (o.get("누락_정보") or [])]
    if st == "추가 확인" and xs and all(re.search("합의|동의", x) for x in xs) and not PAST.search(G[q]["질문"]):
        return "사전 합의 필요"
    return st


cells = collections.defaultdict(dict)  # (실험, 모델) → 문항 → 행
for f in sorted(glob.glob(f"{W}/results/results_*.jsonl")):
    if f.endswith("_models.jsonl"):
        continue
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if r.get("오류") is None and args.exp in r["실험ID"]:
            cells[(r["실험ID"], r["모델ID"])][r["문항번호"]] = r

out = ["# 평가 요약 (자동, 채점.py)", "",
       f"V3 판본이 V2 정답과 다른 문항 {len(CHANGED)}개: {' '.join(sorted(CHANGED))}", "",
       "| 실험 | 모델 | 문항 | 형식OK | 상태 | 주지표 | 주지표+R3 | 판본바뀐것 뺀 주지표+R3 | 잘림 | 평균초 |", "|---|---|---|---|---|---|---|---|---|---|"]
teach = {}
for (e, m), rows in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    n = len(rows)
    ok = st = main = main3 = sub = subn = cut = 0
    secs = []
    for q, r in rows.items():
        o = parse(r)
        cut += r.get("종료_사유") == "length"
        if r.get("응답_시간_초"):
            secs.append(r["응답_시간_초"])
        if o is None:
            continue
        ok += 1
        if "TEACH" in e:
            teach.setdefault(m, {})[q] = o.get("상태")
            continue
        if q not in G:
            continue
        g = G[q]
        v_ok = o.get("적용_판본") == g["적용_판본"]
        st += o.get("상태") == g["정답_상태"]
        main += v_ok and o.get("상태") == g["정답_상태"]
        hit3 = v_ok and r3(q, o) == g["정답_상태"]
        main3 += hit3
        if q not in CHANGED:
            subn += 1
            sub += hit3
    avg = f"{sum(secs) / len(secs):.1f}" if secs else "-"
    if "TEACH" in e:
        out.append(f"| {e} | {m} | {n} | {ok} | - | - | - | - | {cut} | {avg} |")
    else:
        out.append(f"| {e} | {m} | {n} | {ok} | {st} | {main} | {main3} | {sub}/{subn} | {cut} | {avg} |")
if len(teach) >= 2:
    a, b = list(teach)[:2]
    both = set(teach[a]) & set(teach[b])
    same = sum(teach[a][q] == teach[b][q] for q in both)
    out += ["", f"교사 일치: {a} vs {b} — 둘 다 답한 {len(both)}문항 중 상태 같음 {same}",
            "상태 분포 " + " / ".join(f"{m}: {dict(collections.Counter(t.values()).most_common())}" for m, t in teach.items())]
open(args.out, "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
