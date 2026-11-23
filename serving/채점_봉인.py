# 봉인 50 채점기 (봉인 채팅에서 실행. CPU만, 몇 초)
# 사용: python 채점_봉인.py <봉인정답.csv> <results_SEAL_V?.jsonl> <판본_A규칙_SEAL_V?.jsonl> [봉인문항_50.jsonl]
#   정답 CSV에서 쓰는 열: 문항번호, 정답_상태(또는 상태), 적용_판본(또는 판본). 질문 열이 없으면 4번째 인자의 질문 파일로 R3를 계산
# 지표
#   상태      : 상태만 정답
#   주지표    : 상태와 적용_판본이 둘 다 정답
#   주지표+R3 : R3(누락이 합의·동의뿐인 '추가 확인'을 아직 안 한 일이면 '사전 합의 필요'로) 적용 후 주지표
#   규칙이 맞은 문항만 : A 판본 규칙이 정답 판본과 같은 문항에서의 주지표+R3 (모델 실력과 규칙 실력을 나눠 보기 위해)
import csv, json, re, sys, collections

gold_path, res_path, ver_path = sys.argv[1:4]
G = {r["문항번호"]: r for r in csv.DictReader(open(gold_path, encoding="utf-8-sig"))}
col = lambda r, *names: next((r[n] for n in names if n in r), None)
Q = {}
if len(sys.argv) > 4:
    Q = {json.loads(l)["문항번호"]: json.loads(l)["질문"] for l in open(sys.argv[4], encoding="utf-8") if l.strip()}
RULE = {json.loads(l)["문항번호"]: json.loads(l)["판본"] for l in open(ver_path, encoding="utf-8") if l.strip()}
PAST = re.compile(r"(결제했|결제를 했|샀어|샀는데|구매했|냈어|냈는데|보냈어|지불했|이미 [^ ]*(했|샀|냈|보냈|결제)|먹었|다녀왔|했어\. ?(이제|이걸)|처리해도|정산해도|공동비로 (처리|넣)|공동 ?돈으로 (처리|정산))")


def r3(q, o):
    xs = [str(x) for x in (o.get("누락_정보") or [])]
    question = col(G[q], "질문") or Q.get(q, "")
    if o.get("상태") == "추가 확인" and xs and all(re.search("합의|동의", x) for x in xs) and not PAST.search(question):
        return "사전 합의 필요"
    return o.get("상태")


rows = collections.defaultdict(dict)
for line in open(res_path, encoding="utf-8"):
    r = json.loads(line)
    if r.get("오류") is None:
        rows[r["모델ID"]][r["문항번호"]] = r
rule_ok = {q for q in G if RULE.get(q) == col(G[q], "적용_판본", "판본")}
print(f"정답 {len(G)}문항 · A 판본 규칙이 정답과 같은 문항 {len(rule_ok)}개")
print("| 모델 | 형식OK | 상태 | 주지표 | 주지표+R3 | 규칙이 맞은 문항만 주지표+R3 | 평균초 |")
print("|---|---|---|---|---|---|---|")
for m, rs in sorted(rows.items()):
    ok = st = main = main3 = sub = 0
    secs = []
    wrong = []
    for q, g in G.items():
        r = rs.get(q)
        if not r:
            wrong.append(f"{q}(출력 없음)"); continue
        if r.get("응답_시간_초"):
            secs.append(r["응답_시간_초"])
        try:
            o = json.loads(r["출력_원문"])
        except Exception:
            wrong.append(f"{q}(형식 오류)"); continue
        ok += 1
        gs, gv = col(g, "정답_상태", "상태"), col(g, "적용_판본", "판본")
        v = o.get("적용_판본") == gv
        st += o.get("상태") == gs
        main += v and o.get("상태") == gs
        hit = v and r3(q, o) == gs
        main3 += hit
        sub += hit and q in rule_ok
        if not hit:
            wrong.append(f"{q}(상태 {r3(q, o)}/{gs}, 판본 {o.get('적용_판본')}/{gv})")
    avg = f"{sum(secs) / len(secs):.1f}" if secs else "-"
    print(f"| {m} | {ok} | {st} | {main} | {main3} | {sub}/{len(rule_ok)} | {avg} |")
    print(f"  틀린 문항 ({m}): " + "; ".join(wrong))
