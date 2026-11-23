# 공개 100 채점기 (CPU만, 몇 초). B의 채점.py는 상태·판본만 보므로, 공개 100의 여러 유형을 따로 채점한다.
# 사용: python 채점_PUB.py
#   results/results_PUB_V2.jsonl 은 data/공개정답_V2_PUB100.csv 로, results_PUB_V3.jsonl 은 V3 정답지로 채점한다.
# 채점 항목
#   판정 40문항 : 상태 정답 (그대로 / R3 적용 두 가지)
#   판본 13문항 : 판본_채점=예 인 문항만 적용_판본 정답
#   형식 41문항 : 수치 26 · 선택 10 · 예아니오 5 → '답' 칸에서 값을 뽑아 정답과 비교
#   근거 100문항: 필수 근거 쪽 묶음(;로 구분, |는 그중 하나) 중 몇 개를 모델 근거가 짚었는지
#   조회 8 · 서술 11 : 자동 정오 판단 없음 → 근거만 채점하고, 사람이 볼 대조표(CSV)를 만든다
import csv, json, os, re, collections

W = os.path.dirname(os.path.abspath(__file__))
PAST = re.compile(r"(결제했|결제를 했|샀어|샀는데|구매했|냈어|냈는데|보냈어|지불했|이미 [^ ]*(했|샀|냈|보냈|결제)|먹었|다녀왔|했어\. ?(이제|이걸)|처리해도|정산해도|공동비로 (처리|넣)|공동 ?돈으로 (처리|정산))")


def r3(o, question):  # 채점.py와 같은 R3: 누락이 합의·동의뿐인 '추가 확인'을 앞으로의 일이면 '사전 합의 필요'로
    xs = [str(x) for x in (o.get("누락_정보") or [])]
    if o.get("상태") == "추가 확인" and xs and all(re.search("합의|동의", x) for x in xs) and not PAST.search(question):
        return "사전 합의 필요"
    return o.get("상태")


# ── 수치: '2만원', '14,670만 원·1,380만 원', '18시', '12%', '19주', '9500원' ──
NUM = r"(\d[\d,]*(?:\.\d+)?)"


def money_values(text):
    """글 속 금액을 모두 원 단위로 뽑는다. '1억 4,670만 원', '2만원', '9,500원', 단위 없는 숫자도 후보로"""
    t = text.replace(" ", "")
    vals = []
    for m in re.finditer(rf"(?:{NUM}억)?(?:{NUM}만)?{NUM}?원?", t):
        eok, man, won = m.group(1), m.group(2), m.group(3)
        if not (eok or man or won):
            continue
        f = lambda s: float(s.replace(",", "")) if s else 0.0
        v = f(eok) * 1e8 + f(man) * 1e4 + f(won)
        vals.append(v)
        if not eok and not man and won:  # 단위 없는 숫자: '만 원 단위로' 답했을 수 있으니 ×1만도 후보
            vals.append(f(won) * 1e4)
    return vals


def time_values(text):
    vals = []
    for m in re.finditer(r"(오전|오후)?\s*(\d{1,2})\s*(?:시|:(\d{2}))", text):
        h = int(m.group(2)) + (12 if m.group(1) == "오후" and int(m.group(2)) < 12 else 0)
        vals.append(h * 60 + int(m.group(3) or 0))
    return vals


def parse_expected(s):
    """정답 '14,670만 원·1,380만 원' → [('돈', 146700000), ('돈', 13800000)]"""
    out = []
    for part in s.split("·"):
        p = part.strip()
        if p.endswith("시"):
            out.append(("시각", time_values(p)[0]))
        elif p.endswith("%"):
            out.append(("퍼센트", float(p[:-1])))
        elif p.endswith("주"):
            out.append(("주", float(p[:-1].replace(",", ""))))
        else:
            out.append(("돈", money_values(p)[0]))
    return out


def number_ok(expected, answer):
    for kind, v in parse_expected(expected):
        if kind == "돈":
            got = money_values(answer)
        elif kind == "시각":
            got = time_values(answer)
        elif kind == "퍼센트":
            got = [float(x.replace(",", "")) for x in re.findall(rf"{NUM}\s*(?:%|퍼센트)", answer)]
        else:
            got = [float(x.replace(",", "")) for x in re.findall(rf"{NUM}\s*주", answer)]
        if not any(abs(g - v) < 0.5 for g in got):
            return False
    return True


# ── 선택: 'B', 'A,C,D' ──
def choice_ok(expected, answer):
    want = set(re.findall(r"[A-E]", expected))
    a = answer.strip()
    m = re.match(r"^\(?\[?\s*([A-E](?:\s*[,·/및와]\s*[A-E])*)(?![A-Za-z])", a)  # 답 맨 앞의 선택지
    if not m:
        m = re.search(r"(?:정답|답|선택)\s*[은는:]?\s*\(?([A-E](?:\s*[,·/및와]\s*[A-E])*)(?![A-Za-z])", a)
    got = set(re.findall(r"[A-E]", m.group(1))) if m else set(re.findall(r"(?<![A-Za-z])[A-E](?![A-Za-z])", a))
    return got == want


# ── 예/아니오 ──
def yesno_ok(expected, answer):
    a = re.sub(r"^[\s\"'“‘(\[]+", "", answer)
    if re.match(r"아니", a):
        got = "아니오"
    elif re.match(r"(예|네|맞)", a):
        got = "예"
    else:
        got = "아니오" if "아니" in a[:20] else ("예" if re.search(r"(예|네)", a[:20]) else "")
    return got == expected


# ── 근거 쪽 ──
def evidence(required, o):
    pages = {g.get("쪽") for g in (o.get("근거") or []) if g.get("쪽") is not None}
    groups = [g for g in required.split(";") if g.strip()]
    hit = sum(any(int(p) in pages for p in g.split("|") if p.strip().isdigit()) for g in groups)
    return hit, len(groups)


def load(path):
    rows = collections.defaultdict(dict)
    if not os.path.exists(path):
        return rows
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("오류") is None:
            rows[r["모델ID"]][r["문항번호"]] = r
    return rows


out = ["# 공개 100 채점 (채점_PUB.py)", "",
       "| 문서 | 모델 | 형식OK | 판정 상태 /40 | 판정 상태+R3 /40 | 판본 /13 | 수치 /26 | 선택 /10 | 예아니오 /5 | 형식 합 /41 | 근거 쪽 짚은 비율 | 근거 다 짚음 /100 | 잘림 | 평균초 |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for doc in ("V2", "V3"):
    G = {r["문항번호"]: r for r in csv.DictReader(open(f"{W}/data/공개정답_{doc}_PUB100.csv", encoding="utf-8-sig"))}
    for model, rows in sorted(load(f"{W}/results/results_PUB_{doc}.jsonl").items()):
        c = collections.Counter()
        hit_sum = need_sum = 0
        secs = []
        detail = []
        for q, g in G.items():
            r = rows.get(q)
            o = None
            if r:
                c["잘림"] += r.get("종료_사유") == "length"
                if r.get("응답_시간_초"):
                    secs.append(r["응답_시간_초"])
                try:
                    o = json.loads(r["출력_원문"])
                except Exception:
                    o = None
            if o is None:
                detail.append([q, g["유형"], "출력 없음/형식 오류", "", "", "", g["운영진_정답"]])
                continue
            c["형식OK"] += 1
            ans = str(o.get("답") or "")
            mark = ""
            if g["상태_채점"] == "예":
                s0, s3 = o.get("상태") == g["정답_상태"], r3(o, g["질문"]) == g["정답_상태"]
                c["상태"] += s0; c["상태R3"] += s3
                mark = f"상태 {'O' if s0 else 'X'}(R3 {'O' if s3 else 'X'}) 모델={o.get('상태')} 정답={g['정답_상태']}"
            if g["판본_채점"] == "예":
                v = o.get("적용_판본") == g["적용_판본"]
                c["판본"] += v
                mark += f" / 판본 {'O' if v else 'X'} 모델={o.get('적용_판본')} 정답={g['적용_판본']}"
            t = g["유형"]
            if t in ("수치", "선택", "예아니오"):
                ok = {"수치": number_ok, "선택": choice_ok, "예아니오": yesno_ok}[t](g["형식_정답"], ans)
                c[t] += ok
                mark += f"{t} {'O' if ok else 'X'} 정답={g['형식_정답']}"
            h, n = evidence(g["필수_근거_쪽"], o)
            hit_sum += h; need_sum += n; c["근거전부"] += (h == n)
            pages = ",".join(str(x.get("쪽")) for x in (o.get("근거") or []))
            detail.append([q, t, mark.strip(" /"), ans, f"{h}/{n} (모델 쪽 {pages} · 필수 {g['필수_근거_쪽']})", o.get("상태"), g["운영진_정답"]])
        form = c["수치"] + c["선택"] + c["예아니오"]
        avg = f"{sum(secs) / len(secs):.1f}" if secs else "-"
        rate = f"{hit_sum / need_sum:.0%}" if need_sum else "-"
        out.append(f"| {doc} | {model} | {c['형식OK']} | {c['상태']} | {c['상태R3']} | {c['판본']} | {c['수치']} | {c['선택']} | {c['예아니오']} | {form} | {rate} | {c['근거전부']} | {c['잘림']} | {avg} |")
        with open(f"{W}/results/공개100_대조표_{doc}_{model}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["문항번호", "유형", "자동 채점", "모델 답", "근거 쪽", "모델 상태", "운영진 정답"])
            w.writerows(detail)
out += ["", "조회 8 · 서술 11문항은 자동 정오 판단을 하지 않습니다. results/공개100_대조표_*.csv 에서 '모델 답'과 '운영진 정답'을 사람이 비교하세요."]
open(f"{W}/results/공개100_요약.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
