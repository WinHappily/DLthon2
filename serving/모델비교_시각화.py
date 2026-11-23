# 모델 비교 그림 6종 (ver3 낮 결과 + ver4 밤 결과). 결과가 늘어나면 이 칸만 다시 실행하면 된다
import os, re, sys, glob, collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
os.chdir("/workspace/DLthon02/ver4")
sys.path.insert(0, os.getcwd())
try:
    import koreanize_matplotlib  # noqa: F401  한글 글꼴
except ImportError:
    print("[안내] 한글이 네모로 나오면: pip install koreanize-matplotlib 후 다시 실행")
plt.rcParams["axes.unicode_minus"] = False
from run_all import load_jsonl, MODELS

# 1. 채점: score_cell.py를 그대로 쓰되 그 안의 그림은 끄고 결과만 받는다
files = [p for d in ("../ver3/results", "results") for p in sorted(glob.glob(f"{d}/results_*.jsonl")) if not p.endswith("_models.jsonl")]
src = open("score_cell.py", encoding="utf-8").read()
src = re.sub(r"^RESULT_FILES = .*$", f"RESULT_FILES = {files!r}", src, flags=re.M)
src = re.sub(r"^MAIN_EXP = .*$", 'MAIN_EXP = "T2MR_150"', src, flags=re.M)
_show, plt.show = plt.show, (lambda *a, **k: None)
G = {"load_jsonl": load_jsonl, "MODELS": MODELS}
exec(src, G)
plt.show = _show
plt.close("all")
summary, rows, memory, answers = G["summary"], G["rows"], G["memory"], G["answers"]

OUT = "results/그림_비교"
os.makedirs(OUT, exist_ok=True)
E, BASE = "T2MR_150", "qwen3-32b-awq"
FAM = [("gemma4", "#1f77b4"), ("qwen3.8", "#d62728"), ("qwen3.5", "#ff7f0e"), ("qwen3-", "#e377c2"),
       ("exaone3.5", "#2ca02c"), ("exaone4.0", "#17becf"), ("kanana", "#9467bd")]
def fam(m):
    for k, c in FAM:
        if m.startswith(k):
            return k.rstrip("-"), c
    return "기타", "gray"
def prec(m):
    return "8비트" if "fp8" in m else "4비트" if ("awq" in m or "qat" in m) else "16비트"
MARK = {"16비트": "o", "4비트": "s", "8비트": "^"}
def pct(e, m):
    s = summary.get((e, m))
    return s["주지표_정답률"] if s else np.nan
def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    plt.show()
floor = np.mean([a["정답_상태"] == "가능" for q, a in answers.items() if q.startswith("D")])
by_mem = lambda ms: sorted(ms, key=lambda m: memory.get(m, 999))

# 그림 1: 메모리 × 정답률 (어디까지 줄여도 되나)
ms = by_mem({m for (e, m) in summary if e == E and m in memory})
fig, ax = plt.subplots(figsize=(11, 6.5))
for m in ms:
    s = summary[(E, m)]
    x, y = memory[m], s["주지표_정답률"]
    ax.errorbar(x, y, yerr=[[y - s["구간_하한"]], [s["구간_상한"] - y]], fmt=MARK[prec(m)], color=fam(m)[1], ms=8, capsize=3, alpha=.9)
    ax.annotate(m, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
b = summary.get((E, BASE))
if b:
    ax.axhline(b["주지표_정답률"], color="gray", lw=1)
    ax.axhline((b["주지표_정답"] - 3) / b["문항_수"], color="gray", ls="--", lw=1)
ax.axhline(floor, color="red", ls=":", lw=1)
fams = [Line2D([], [], color=c, marker="o", ls="", label=k.rstrip("-")) for k, c in FAM if any(fam(m)[1] == c for m in ms)]
precs = [Line2D([], [], color="k", marker=v, ls="", label=k) for k, v in MARK.items()]
lines = [Line2D([], [], color="gray", lw=1, label=f"기준선 {BASE}"), Line2D([], [], color="gray", ls="--", lw=1, label="기준선 − 3문항"),
         Line2D([], [], color="red", ls=":", lw=1, label="바닥(전부 '가능')")]
ax.legend(handles=fams + precs + lines, loc="lower right", fontsize=8, ncol=2)
ax.set(xlabel="가중치 메모리 (GiB)", ylabel="주지표 정답률 (상태+판본)", ylim=(0, 1.02),
       title=f"{E}: 모델 크기와 정답률 (막대 = 상황 단위 95% 구간)")
save(fig, "1_메모리_정답률")

# 그림 2: 모델별 세 조건 (오라클 / 최종 시스템 / 잡음)
ms = by_mem({m for (e, m) in summary if e in ("O1p_150", E, "T2MR_N90")})
fig, ax = plt.subplots(figsize=(10, 0.45 * len(ms) + 1.5))
y, h = np.arange(len(ms)), 0.27
for i, (e, lab, c) in enumerate([("O1p_150", "O1p (오라클, 판본은 모델이)", "#bbbbbb"), (E, "T2MR (최종 시스템)", "#1f77b4"), ("T2MR_N90", "T2MR 잡음 90", "#ff7f0e")]):
    v = [pct(e, m) for m in ms]
    ax.barh(y + (i - 1) * h, v, h, label=lab, color=c)
    for yy, vv in zip(y + (i - 1) * h, v):
        if not np.isnan(vv):
            ax.text(vv + .005, yy, f"{vv:.0%}", va="center", fontsize=7)
ax.set_yticks(y, [f"{m} ({memory.get(m, float('nan')):.1f}GiB)" for m in ms], fontsize=8)
ax.invert_yaxis()
ax.axvline(floor, color="red", ls=":", lw=1)
ax.set(xlim=(0, 1.08), xlabel="주지표 정답률", title="모델별 비교 (메모리 작은 순)")
ax.legend(loc="lower right", fontsize=8)
save(fig, "2_모델별_세조건")

# 그림 3: 코드 개입 계단 (소형 4개: 검색만 → +메모 → +판본 고정, 오른쪽은 오라클+판본 고정)
small = by_mem({m for (e, m) in summary if e == "T2_150"})
steps = [("T2", "검색만"), ("T2M", "+판정 메모"), ("T2MR", "+판본 고정"), ("O1pR", "오라클\n+판본 고정")]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, suf in zip(axs, ("150", "N90")):
    for m in small:
        v = [pct(f"{k}_{suf}", m) for k, _ in steps]
        ax.plot(range(3), v[:3], marker="o", color=fam(m)[1], label=m)
        ax.plot(3, v[3], marker="D", mfc="white", color=fam(m)[1])
        ax.annotate(f"{v[2]:.0%}", (2, v[2]), xytext=(4, 0), textcoords="offset points", fontsize=8, va="center")
    ax.set_xticks(range(4), [lab for _, lab in steps])
    ax.axvline(2.5, color="lightgray", lw=1)
    ax.axhline(floor if suf == "150" else np.mean([a["정답_상태"] == "가능" for q, a in answers.items() if q.startswith("N")]), color="red", ls=":", lw=1)
    top = np.nanmax([pct(f"{k}_{suf}", m) for m in small for k, _ in steps] + [0.6])
    ax.set(title=f"코드 개입 효과 ({'개발 150' if suf == '150' else '잡음 90'})", ylim=(0, min(1.0, top + 0.1)))
axs[0].set_ylabel("주지표 정답률")
axs[0].legend(fontsize=8, loc="upper left")
save(fig, "3_코드개입_계단")

# 그림 4: 예측 상태 분포 (남은 병목: '추가 확인' 쏠림, '사전 합의 필요' 누락)
S = ["가능", "사전 합의 필요", "공동 처리 불가", "추가 확인", "규정에 없음", "(깨짐)"]
SC = ["#4c9f70", "#f2a541", "#d1495b", "#6c8ebf", "#9e9e9e", "#333333"]
rs = [r for r in rows if r["실험ID"] == E]
ms = by_mem({r["모델ID"] for r in rs})
gold = collections.Counter(r["정답_상태"] for r in rs if r["모델ID"] == ms[0])
labels = ["(정답)"] + ms
dist = [gold] + [collections.Counter(r["예측_상태"] for r in rs if r["모델ID"] == m) for m in ms]
fig, ax = plt.subplots(figsize=(11, 0.42 * len(labels) + 1.5))
left = np.zeros(len(labels))
for st, c in zip(S, SC):
    v = np.array([d[st] for d in dist], float)
    ax.barh(range(len(labels)), v, left=left, color=c, label=st)
    for i, (l, vv) in enumerate(zip(left, v)):
        if vv >= 8:
            ax.text(l + vv / 2, i, int(vv), ha="center", va="center", fontsize=7, color="white")
    left += v
ax.set_yticks(range(len(labels)), labels, fontsize=8)
ax.invert_yaxis()
ax.set(xlabel="문항 수", title=f"{E}: 모델이 고른 상태 분포 (맨 위가 정답 분포)")
ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
save(fig, "4_상태분포")

# 그림 5: 4비트 vs 8비트 (같은 모델 세 쌍)
pairs = [(a, b) for a, b in [("qwen3-32b-awq", "qwen3-32b-fp8"), ("qwen3.8-27b-awq", "qwen3.8-27b-fp8"), ("gemma4-31b-qat", "gemma4-31b-fp8")]
         if (E, a) in summary and (E, b) in summary]
if pairs:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(pairs))
    for j, (lab, c) in enumerate([("4비트", "#1f77b4"), ("8비트", "#aec7e8")]):
        v = [pct(E, p[j]) for p in pairs]
        ax.bar(x + (j - .5) * .38, v, .38, color=c, label=lab)
        for xx, vv, p in zip(x + (j - .5) * .38, v, pairs):
            ax.text(xx, vv + .01, f"{vv:.0%}\n{memory.get(p[j], float('nan')):.1f}GiB", ha="center", fontsize=8)
    ax.set_xticks(x, [p[0].split("-awq")[0].split("-qat")[0] for p in pairs])
    ax.set(ylim=(0, 1.1), ylabel="주지표 정답률", title=f"{E}: 4비트 vs 8비트 (막대 위: 정답률, 메모리)")
    ax.legend()
    save(fig, "5_4비트_8비트")

# 그림 6: 응답 시간 (동시 요청 1, 앞 30문항, T2MR 설정)
lat, ttft = collections.defaultdict(list), collections.defaultdict(list)
for r in load_jsonl("results/results_LAT_150.jsonl"):
    if r["오류"] is None:
        lat[r["모델ID"]].append(r["응답_시간_초"]); ttft[r["모델ID"]].append(r["첫토큰_시간_초"])
ms = by_mem(lat)
if ms:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(ms))
    ax.bar(x, [np.mean(lat[m]) for m in ms], color=[fam(m)[1] for m in ms], label="전체 응답")
    ax.bar(x, [np.mean(ttft[m]) for m in ms], color="black", alpha=.35, width=.5, label="첫 토큰까지")
    for xx, m in zip(x, ms):
        ax.text(xx, np.mean(lat[m]) + .3, f"{np.mean(lat[m]):.1f}초\n{pct(E, m):.0%}" if not np.isnan(pct(E, m)) else f"{np.mean(lat[m]):.1f}초", ha="center", fontsize=8)
    ax.set_xticks(x, [f"{m}\n{memory.get(m, float('nan')):.1f}GiB" for m in ms], fontsize=8, rotation=20)
    ax.set(ylabel="초 (30문항 평균)", title="응답 시간 (막대 위: 평균 초, T2MR 정답률)")
    ax.legend()
    save(fig, "6_응답시간")
print("저장:", OUT)
