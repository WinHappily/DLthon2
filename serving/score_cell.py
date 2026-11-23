# 7. 자동 채점 · 태그별 분석 · 발표용 그림 (앞의 4번 칸까지 실행한 뒤 실행)
import collections
import csv
import glob
import os
import json
import re

import matplotlib.pyplot as plt
import numpy as np

try:
    import koreanize_matplotlib  # noqa: F401  그림의 한글 글꼴
except ImportError:
    print("[안내] 그림의 한글이 네모로 나오면: pip install koreanize-matplotlib 후 이 칸을 다시 실행")

RESULT_FILES = ["results/results_O1p_150.jsonl", "results/results_T2M_150.jsonl"]  # 채점할 결과 파일을 모두 적는다
ANSWERS = "data/개발정답_150_N90_F30.csv"  # 개발 150 + 잡음 90 + 형식 30 (형식은 상태·판본 참고용)
OUT_DIR = "results/채점"             # 채점 CSV와 그림이 저장되는 폴더
MAIN_EXP = "O1p_150"                 # 태그 히트맵·혼동표를 만들 실험
COMPARE = ("O1p_150", "T2M_150")     # 그림 2: 두 실험이 다 있을 때만 그림
BASELINE = "qwen3-32b-awq"           # 32B 기준선
FOCUS = "qwen3-4b-instruct-2507"     # 그림 4에서 기준선과 나란히 볼 모델
ALLOWED = {"제3조", "표 06", "제7조", "제8조", "표 07", "부칙 제1조", "표 53", "제9조"}  # 허용 근거(오라클 공통 조각과 같게): 초과 인용에서 뺀다
BOOT = 2000                          # 신뢰구간 부트스트랩 횟수
STATES = ["가능", "사전 합의 필요", "공동 처리 불가", "추가 확인", "규정에 없음", "(깨짐)"]

SRC = re.compile(r"(부칙\s*)?제\s*(\d+)\s*조|표\s*(\d+)")


def sources(text):
    """'표 7', '표07', '제 23 조', '부칙제3조' 같은 표기를 '표 07', '제23조', '부칙 제3조'로 맞춘다."""
    out = []
    for m in SRC.finditer(text or ""):
        out.append(f"표 {int(m.group(3)):02d}" if m.group(3)
                   else ("부칙 " if m.group(1) else "") + f"제{int(m.group(2))}조")
    return out


def tags_of(a):
    groups = [g for g in a["필수_근거"].split(";") if g]
    t = []
    if a["개정_항목"] not in ("", "해당 없음"):
        t.append("판본 판단")
    if a["기준일"] in ("2026-09-30", "2026-10-01"):
        t.append("경계 날짜")
    if "기준일" in a["유형"]:
        t.append("기준일 함정")
    if "경과조치" in a["유형"]:
        t.append("경과조치")
    if any(all(s.startswith("표") for s in sources(g)) for g in groups):
        t.append("표 필수")
    if len(groups) > 1:
        t.append("복합")
    if a["정답_상태"] == "추가 확인":
        t.append("정보 부족")
    if a["정답_상태"] == "규정에 없음":
        t.append("규정에 없음")
    t += [x.strip() for x in a.get("태그", "").split(",") if x.strip()]  # 사람이 붙인 태그 칸이 있으면 더한다
    return t or ["단순"]


def score(a, out):
    if not isinstance(out, dict):
        return {"예측_상태": "(깨짐)", "예측_판본": "(깨짐)", "상태": 0, "판본": 0, "주지표": 0,
                "근거_충족": None, "근거_재현율": None, "초과_인용": None}
    cited = {s for e in out.get("근거") or [] if isinstance(e, dict) for s in sources(str(e.get("출처", "")))}
    st, ver = int(out.get("상태") == a["정답_상태"]), int(out.get("적용_판본") == a["적용_판본"])
    groups = [[x for alt in g.split("|") for x in sources(alt)[:1]] for g in a["필수_근거"].split(";") if g]
    full = recall = extra = None
    if a["근거_채점_대상"] == "예" and groups:
        hits = [any(x in cited for x in g) for g in groups]
        full, recall = int(all(hits)), sum(hits) / len(groups)
        extra = len(cited - set(sources(a["필수_근거"] + " " + a["보조_근거_행"])) - ALLOWED)
    return {"예측_상태": out.get("상태"), "예측_판본": out.get("적용_판본"), "상태": st, "판본": ver,
            "주지표": st * ver, "근거_충족": full, "근거_재현율": recall, "초과_인용": extra}


def boot_ci(vals, clusters):
    """상황 단위로 다시 뽑는 부트스트랩: 변형 문항끼리 묶어 실질 표본 수를 반영한다."""
    by = collections.defaultdict(list)
    for v, c in zip(vals, clusters):
        by[c].append(v)
    keys, rng = list(by), np.random.default_rng(0)
    stats = [np.mean([v for i in rng.integers(0, len(keys), len(keys)) for v in by[keys[i]]]) for _ in range(BOOT)]
    return np.percentile(stats, [2.5, 97.5])


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


answers = {a["문항번호"]: a for a in csv.DictReader(open(ANSWERS, encoding="utf-8-sig"))}
for a in answers.values():
    a["상황"] = a.get("상황번호") or a["쌍번호"] or a["문항번호"]
    a["태그"] = tags_of(a)

latest = {}
for path in RESULT_FILES:
    for r in load_jsonl(path):
        if r["문항번호"] in answers:
            latest[(r["실험ID"], r["모델ID"], r["문항번호"])] = r  # 같은 문항을 여러 번 돌렸으면 마지막 줄

rows = []
for (exp, mid, qid), r in latest.items():
    a = answers[qid]
    try:
        out = json.loads(r["출력_원문"]) if r["오류"] is None and r["출력_원문"] else None
    except json.JSONDecodeError:
        out = None
    rows.append({"실험ID": exp, "모델ID": mid, "문항번호": qid, "상황번호": a["상황"], "태그": ", ".join(a["태그"]),
                 "정답_상태": a["정답_상태"], "정답_판본": a["적용_판본"], **score(a, out),
                 "응답_시간_초": r.get("응답_시간_초")})

exps = sorted({r["실험ID"] for r in rows})
order = [m for m in MODELS if any(r["모델ID"] == m for r in rows)]
order += sorted({r["모델ID"] for r in rows} - set(order))
print(f"채점: 실험 {exps}, 모델 {len(order)}개, 문항 {len(answers)}개, 상황 {len({a['상황'] for a in answers.values()})}개")
print(f"바닥 점수(모든 문항에 '가능'이라고만 답할 때의 상태 정답률): "
      f"{sum(a['정답_상태'] == '가능' for a in answers.values()) / len(answers):.0%}")


def write_csv(path, dict_rows):
    path = os.path.join(OUT_DIR, path)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict_rows[0]))
        w.writeheader()
        w.writerows(dict_rows)
    print("저장:", path)


write_csv("채점_문항별.csv", rows)

# 모델 요약 (실험마다)
summary = {}
for exp in exps:
    base = [r for r in rows if r["실험ID"] == exp and r["모델ID"] == BASELINE]
    for mid in order:
        rs = [r for r in rows if r["실험ID"] == exp and r["모델ID"] == mid]
        if not rs:
            continue
        lo, hi = boot_ci([r["주지표"] for r in rs], [r["상황번호"] for r in rs])
        pairs = collections.defaultdict(list)
        for r in rs:
            if answers[r["문항번호"]]["쌍번호"]:
                pairs[answers[r["문항번호"]]["쌍번호"]].append(r["주지표"])
        k = sum(r["주지표"] for r in rs)
        s = {"실험ID": exp, "모델ID": mid, "문항_수": len(rs), "주지표_정답": k,
             "주지표_정답률": round(k / len(rs), 3), "구간_하한": round(lo, 3), "구간_상한": round(hi, 3),
             "상태_정답률": mean([r["상태"] for r in rs]), "판본_정답률": mean([r["판본"] for r in rs]),
             "근거_충족률": mean([r["근거_충족"] for r in rs]), "근거_재현율": mean([r["근거_재현율"] for r in rs]),
             "평균_초과_인용": mean([r["초과_인용"] for r in rs]),
             "쌍_일관성": mean([int(all(v)) for v in pairs.values() if len(v) == 2]),
             "JSON_깨짐": sum(r["예측_상태"] == "(깨짐)" for r in rs),
             "기준선_대비_문항": (k - sum(r["주지표"] for r in base)) if len(base) == len(rs) else None}
        summary[(exp, mid)] = s
write_csv("채점_모델요약.csv", list(summary.values()))

# 태그별 정답률과 혼동표 (MAIN_EXP)
main = [r for r in rows if r["실험ID"] == MAIN_EXP]
all_tags = sorted({t for a in answers.values() for t in a["태그"]}, key=lambda t: -sum(t in a["태그"] for a in answers.values()))
tag_rows, cell = [], {}
for mid in order:
    for t in all_tags:
        rs = [r for r in main if r["모델ID"] == mid and t in answers[r["문항번호"]]["태그"]]
        if rs:
            k, n = sum(r["주지표"] for r in rs), len(rs)
            lo, hi = wilson(k, n)
            cell[(mid, t)] = (k, n)
            tag_rows.append({"모델ID": mid, "태그": t, "문항_수": n, "정답": k, "정답률": round(k / n, 3),
                             "구간_하한": round(lo, 3), "구간_상한": round(hi, 3)})
if tag_rows:
    write_csv(f"채점_{MAIN_EXP}_태그별.csv", tag_rows)

conf = {}
for mid in order:
    c = collections.Counter((r["정답_상태"], r["예측_상태"]) for r in main if r["모델ID"] == mid)
    conf[mid] = np.array([[c[(t, p)] for p in STATES] for t in STATES[:-1]])
if main:
    write_csv(f"채점_{MAIN_EXP}_혼동표.csv",
              [{"모델ID": m, "정답_상태": t, **{f"예측_{p}": int(conf[m][i][j]) for j, p in enumerate(STATES)}}
               for m in order if m in conf for i, t in enumerate(STATES[:-1])])

# ---------- 그림 ----------
memory = {}
for path in {m for p in RESULT_FILES for m in glob.glob(os.path.join(os.path.dirname(p), "*_models.jsonl"))}:
    for r in load_jsonl(path):
        if r.get("가중치_메모리_GiB"):
            memory[r["모델ID"]] = r["가중치_메모리_GiB"]

# 그림 1: 정답률 × 가중치 메모리
pts = [summary[(MAIN_EXP, m)] for m in order if (MAIN_EXP, m) in summary and m in memory]
if pts:
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in pts:
        x, y = memory[s["모델ID"]], s["주지표_정답률"]
        ax.errorbar(x, y, yerr=[[y - s["구간_하한"]], [s["구간_상한"] - y]], fmt="o", capsize=4)
        ax.annotate(s["모델ID"], (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    b = summary.get((MAIN_EXP, BASELINE))
    if b:
        ax.axhline(b["주지표_정답률"], color="gray", lw=1, label=f"기준선 {BASELINE}")
        ax.axhline((b["주지표_정답"] - 3) / b["문항_수"], color="gray", ls="--", lw=1, label="기준선 − 3문항")
        ax.legend(loc="lower right")
    ax.set(xlabel="가중치 메모리 (GiB)", ylabel="정답률 (상태+판본)", ylim=(0, 1.05),
           title=f"{MAIN_EXP}: 모델 크기와 정답률 (막대 = 상황 단위 95% 구간)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f"그림1_{MAIN_EXP}_정답률_메모리.png"), dpi=150, bbox_inches="tight"); plt.show()

# 그림 2: 오라클 vs T2
both = [m for m in order if all((e, m) in summary for e in COMPARE)]
if both:
    fig, ax = plt.subplots(figsize=(max(6, len(both) * 1.2), 4.5))
    xs = np.arange(len(both))
    for i, e in enumerate(COMPARE):
        ys = [summary[(e, m)]["주지표_정답률"] for m in both]
        err = [[y - summary[(e, m)]["구간_하한"] for m, y in zip(both, ys)],
               [summary[(e, m)]["구간_상한"] - y for m, y in zip(both, ys)]]
        ax.bar(xs + (i - 0.5) * 0.4, ys, 0.4, yerr=err, capsize=3, label=e)
    ax.set_xticks(xs, both, rotation=30, ha="right")
    ax.set(ylabel="정답률 (상태+판본)", ylim=(0, 1.05), title=f"{COMPARE[0]} vs {COMPARE[1]}: 차이 = 검색 탓")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "그림2_오라클_T2.png"), dpi=150, bbox_inches="tight"); plt.show()
else:
    print(f"[그림 2 생략] {COMPARE} 두 실험 결과가 다 있어야 그립니다")

# 그림 3: 모델 × 태그 히트맵
if cell:
    grid = np.array([[cell[(m, t)][0] / cell[(m, t)][1] if (m, t) in cell else np.nan for t in all_tags] for m in order])
    fig, ax = plt.subplots(figsize=(1.1 * len(all_tags) + 3, 0.6 * len(order) + 2))
    ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i, m in enumerate(order):
        for j, t in enumerate(all_tags):
            if (m, t) in cell:
                k, n = cell[(m, t)]
                ax.text(j, i, f"{k / n:.0%}\n{k}/{n}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(all_tags)), all_tags, rotation=30, ha="right")
    ax.set_yticks(range(len(order)), order)
    ax.set_title(f"{MAIN_EXP}: 유형별 정답률 (칸 = 정답/문항 수)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f"그림3_{MAIN_EXP}_유형_히트맵.png"), dpi=150, bbox_inches="tight"); plt.show()

# 그림 4: 상태 혼동표 (기준선 vs FOCUS)
pair = [m for m in (BASELINE, FOCUS) if m in conf and conf[m].sum()]
if pair:
    fig, axes = plt.subplots(1, len(pair), figsize=(6.5 * len(pair), 5.5), squeeze=False)
    for ax, m in zip(axes[0], pair):
        ax.imshow(conf[m], cmap="Blues")
        for i in range(len(STATES) - 1):
            for j in range(len(STATES)):
                if conf[m][i][j]:
                    ax.text(j, i, int(conf[m][i][j]), ha="center", va="center")
        ax.set_xticks(range(len(STATES)), STATES, rotation=30, ha="right")
        ax.set_yticks(range(len(STATES) - 1), STATES[:-1])
        ax.set(xlabel="모델이 답한 상태", ylabel="정답 상태", title=m)
    fig.suptitle(f"{MAIN_EXP}: 상태 혼동표 (대각선 = 맞힘)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f"그림4_{MAIN_EXP}_혼동표.png"), dpi=150, bbox_inches="tight"); plt.show()
