# 결과 파일(run_all 출력)을 제출용 CSV로 펼친다. 운영진 서식의 열 이름·순서에 맞게 COLUMNS만 고치면 된다
# 사용: python to_submission.py <results_XXX.jsonl> <모델ID> <제출.csv> [질문.jsonl]
#   질문 파일을 주면 R3(누락이 합의·동의뿐인 '추가 확인'을 앞으로의 일이면 '사전 합의 필요'로)를 적용한다
import csv, json, re, sys
COLUMNS = {  # 제출 열 이름: 출력 JSON에서 가져올 값
    "문항번호": lambda q, o: q,
    "판정": lambda q, o: o.get("상태"),
    "답": lambda q, o: o.get("답"),
    "적용_판본": lambda q, o: o.get("적용_판본"),
    "기준일": lambda q, o: o.get("기준일"),
    "근거": lambda q, o: "; ".join(f"{g.get('출처')}{' ' + g['행'] if g.get('행') else ''}{' (' + str(g['쪽']) + '쪽)' if g.get('쪽') else ''}" for g in (o.get("근거") or [])),
    "다음_행동": lambda q, o: o.get("다음_행동"),
    "확인_질문": lambda q, o: o.get("추가_확인_질문"),
}
src, model, out = sys.argv[1:4]
QT = {json.loads(l)["문항번호"]: json.loads(l)["질문"] for l in open(sys.argv[4], encoding="utf-8")} if len(sys.argv) > 4 else None
PAST = re.compile(r"(결제했|결제를 했|샀어|샀는데|구매했|냈어|냈는데|보냈어|지불했|이미 [^ ]*(했|샀|냈|보냈|결제)|먹었|다녀왔|했어\. ?(이제|이걸)|처리해도|정산해도|공동비로 (처리|넣)|공동 ?돈으로 (처리|정산))")


def r3(q, o):
    xs = [str(x) for x in (o.get("누락_정보") or [])]
    if QT and o.get("상태") == "추가 확인" and xs and all(re.search("합의|동의", x) for x in xs) and not PAST.search(QT.get(q, "")):
        o["상태"] = "사전 합의 필요"
    return o
rows = {}
for line in open(src, encoding="utf-8"):
    r = json.loads(line)
    if r["모델ID"] == model and r["오류"] is None:
        rows[r["문항번호"]] = r
bad = []
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(COLUMNS)
    for q in sorted(rows):
        try:
            o = r3(q, json.loads(rows[q]["출력_원문"]))
        except Exception:
            o, _ = {}, bad.append(q)
        w.writerow([fn(q, o) for fn in COLUMNS.values()])
print(f"{out}: {len(rows)}문항, JSON이 깨진 문항 {bad or '없음'}")
