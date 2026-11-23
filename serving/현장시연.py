#!/usr/bin/env python3
"""현장 시연: 질문을 그 자리에서 받아 최종 구성(검색 + 판정 메모 + 판본 고정 + 지시문 p3 + R3)으로 판정한다.

평가 때와 같은 코드를 그대로 쓴다.
  검색 · 판정 메모 · 판본 결정 = ../retrieval 의 rag (A 코드, 판본 규칙 g3.3)
  지시문 · 출력 형식 · 판본 고정 · 모델 호출 = run_all.py (B 코드)
  R3 = to_submission.py 와 같은 규칙
다른 점은 하나: 미리 만들어 둔 검색 결과 파일 대신, 질문이 들어오면 그 자리에서 검색한다.

처음 한 번 (질문 벡터용 bge-m3 색인 만들기, GPU가 비어 있으면 1분, CPU면 5~10분):
  python 현장시연.py --build-index
시연:
  python 현장시연.py --model gemma4-12b-qat
  질문>  (질문 문장을 그대로 붙여 넣는다)
  명령:  :1 :2 :3 미리 정한 시연 질문 · :model <모델ID> 모델 바꾸기 · :docs V3 문서 고르기 · :q 끝
모델 서버(vLLM)는 이 스크립트가 띄우고, 끝나도 켜 둔다(다음 실행에서 바로 씀). 끄기: pkill -f "vllm serve"
"""
import os

# 질문 벡터(bge-m3)는 CPU로 계산한다. GPU는 vLLM이 90%를 잡기 때문. 모델 서버를 띄울 때는 원래 값으로 돌려준다
_GPU_ENV = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.join(HERE, "..", "retrieval")
sys.path.insert(0, HERE)
sys.path.insert(0, RAG_DIR)

import numpy as np
import requests

import run_all as R
from rag.brief import brief_text, build_brief
from rag.common import ROOT
from rag.search import Searcher

EMB_MODEL = "BAAI/bge-m3"
PORT = 8000
MAX_TOKENS = 2048

# 미리 정한 시연 질문 (슬라이드 6 · 21). 1 · 3은 개발 문항 그대로(D002 · D117), 2는 개정 이후 사건을 새로 쓴 것
PRESETS = {
    "1": "시아버지 생신이라 식사 15만 원은 내 카드로, 선물 8만 원은 감석 카드로 결제했어. 첫 결제는 2026-10-15였고, 결제 전에 둘이 범위와 상한을 합의해 뒀어. 이번 달 가족행사 예산도 남아 있어. 공동비로 처리할 수 있어?",
    "2": "2026-12-05부터 친구가 우리 집에서 2박 하기로 했어. 시작 60시간 전에 감석에게 날짜와 인원을 알려 뒀어. 따로 합의도 받아야 해?",
    "3": None,  # 아래에서 개발 문항 D117 원문을 읽어 채운다
}

# ---------------------------------------------------------------- 질문 벡터 (bge-m3)

class Encoder:
    """sentence-transformers 없이 bge-m3 문장 벡터를 만든다: CLS 위치 벡터 + 길이 1로 정규화 (bge-m3의 문장 벡터 방식)"""

    def __init__(self, name=EMB_MODEL, device="cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()

    def encode(self, texts, normalize_embeddings=True, batch_size=8):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                b = self.tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=8192, return_tensors="pt").to(self.device)
                v = self.model(**b).last_hidden_state[:, 0]
                if normalize_embeddings:
                    v = self.torch.nn.functional.normalize(v, dim=-1)
                out.append(v.float().cpu().numpy())
        return np.concatenate(out)


def build_index(docs, device):
    from rag.index import build_chunks
    enc = Encoder(device=device)
    for doc in docs:
        d = ROOT / "data" / doc
        chunks = build_chunks(d)
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        t0 = time.time()
        emb = enc.encode([c["text"] for c in chunks], batch_size=4)
        np.save(d / "emb.npy", emb.astype("float32"))
        (d / "emb_meta.json").write_text(json.dumps({"model": EMB_MODEL, "dim": int(emb.shape[1]), "ids": [c["id"] for c in chunks]},
                                                    ensure_ascii=False), encoding="utf-8")
        print(f"[{doc}] 조각 {len(chunks)}개 벡터 {emb.shape} {time.time() - t0:.0f}초 -> {d / 'emb.npy'}")


def load_searcher(doc, enc):
    s = Searcher(doc)  # sentence-transformers가 없으면 BM25만 켠 상태로 만들어진다
    emb = ROOT / "data" / doc / "emb.npy"
    if emb.exists():
        s.dense = {"model": enc, "emb": np.load(emb)}
    else:
        print(f"[경고] {doc} 벡터 색인이 없어 키워드 검색만 씁니다. 먼저 python 현장시연.py --build-index")
    return s

# ---------------------------------------------------------------- 검색 결과 → 평가 때와 같은 입력 (convert_a.py와 같은 변환)

TO_TEAM = {"추가 확인": "미정", "규정 범위 밖": "해당 없음"}


def two_digit(name):
    return re.sub(r"표\s*(\d+)", lambda m: f"표 {int(m.group(1)):02d}", name)


def to_chunks(hits):
    return [{"출처": two_digit("·".join([str(x) for x in (h.get("articles") or [])] + [str(x) for x in (h.get("tables") or [])])) or f"{h['page']}쪽",
             "쪽": h["page"], "본문": h["text"],
             "경로": "·".join(map(str, h["why"])) if isinstance(h.get("why"), list) else (h.get("why") or "검색")} for h in hits]


def to_version(v):
    raw = v.get("label") or v.get("applied_version") or v.get("status")
    team = TO_TEAM.get(v.get("team_label") or raw, v.get("team_label") or raw)
    return {"판본": team, "규칙_표기": v.get("label_raw") or raw, "항목": v.get("item"), "기준일": v.get("basis_date"), "이유": v.get("reason")}

# ---------------------------------------------------------------- R3 (to_submission.py와 같은 규칙)

PAST = re.compile(r"(결제했|결제를 했|샀어|샀는데|구매했|냈어|냈는데|보냈어|지불했|이미 [^ ]*(했|샀|냈|보냈|결제)|먹었|다녀왔|했어\. ?(이제|이걸)|처리해도|정산해도|공동비로 (처리|넣)|공동 ?돈으로 (처리|정산))")


def r3(question, o):
    xs = [str(x) for x in (o.get("누락_정보") or [])]
    if o.get("상태") == "추가 확인" and xs and all(re.search("합의|동의", x) for x in xs) and not PAST.search(question):
        return "사전 합의 필요"
    return None

# ---------------------------------------------------------------- 모델 서버

def served_model():
    try:
        return requests.get(f"http://localhost:{PORT}/v1/models", timeout=5).json()["data"][0]["id"]
    except Exception:
        return None


def ensure_server(mid):
    if subprocess.run(["pgrep", "-f", "run_all.py"], capture_output=True).stdout:
        print("다른 실험(run_all.py)이 GPU를 쓰는 중입니다. 끝난 뒤 다시 실행하세요")
        return False
    cur = served_model()
    if cur == mid:
        return True
    if cur or subprocess.run(["pgrep", "-f", "vllm serve"], capture_output=True).stdout:
        print(f"켜져 있던 서버({cur or '응답 없음'})를 끕니다…")
        subprocess.run(["pkill", "-f", "vllm serve"])
        R.wait_gpu_free(180)
    path = os.path.join(R.MODELS_DIR, mid)
    if not os.path.isdir(path):
        print(f"모델 폴더가 없습니다: {path}")
        return False
    env = dict(os.environ)
    if _GPU_ENV is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = _GPU_ENV
    os.makedirs(os.path.join(HERE, "results", "logs"), exist_ok=True)
    log_path = os.path.join(HERE, "results", "logs", f"현장시연_{mid}.log")
    for extra in R.MODELS.get(mid, [[]]):
        cmd = ["vllm", "serve", path, "--served-model-name", mid, "--port", str(PORT), *R.COMMON_ARGS, *extra]
        print(f"[{mid}] 서버 시작 중 (1~3분)", end="", flush=True)
        log = open(log_path, "a", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        t0 = time.time()
        while time.time() - t0 < 900 and proc.poll() is None:
            if R.server_alive(PORT):
                print(f" 준비됨 ({time.time() - t0:.0f}초)")
                return True
            print(".", end="", flush=True)
            time.sleep(5)
        print(" 실패")
        subprocess.run(["pkill", "-f", "vllm serve"])
        R.wait_gpu_free(180)
    print(f"서버를 띄우지 못했습니다. 로그: {log_path}")
    return False

# ---------------------------------------------------------------- 판정 한 번

B, D, G, Y, X = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[0m"


def prompts(doc):
    system_t = open(os.path.join(HERE, f"p3_{doc}.txt"), encoding="utf-8").read()
    system = system_t.format(examples=R.format_examples(os.path.join(HERE, "예시_p3.json")))
    opt = "options_v3.json" if doc == "V3" else "options_v2.json"
    schema = R.build_schema(json.load(open(os.path.join(HERE, opt), encoding="utf-8")))
    return system, schema


def judge(question, doc, searcher, mid, cache, log):
    t0 = time.perf_counter()
    res = searcher.retrieve(question, k=5, max_refs=4)
    memo = brief_text(build_brief(searcher, res))
    chunks, v = to_chunks(res["hits"]), to_version(res["version"])
    t_ret = time.perf_counter() - t0

    print(f"\n{B}━━ {doc} 문서 ━━{X}")
    print(f"{B}① 검색{X} {D}({t_ret:.1f}초, 조각 {len(chunks)}개){X}")
    for c in chunks:
        print(f"   {c['출처']:<28} {c['쪽']:>3}쪽  {D}{c['경로'][:40]}{X}")
    print(f"{B}② 판정 메모{X}")
    for line in (memo or "(없음)").splitlines()[:8]:
        print(f"   {D}{line[:110]}{X}")
    print(f"{B}③ 판본 (코드가 결정){X}  {G}{v['판본']}{X}  {D}{(v['이유'] or '')[:100]}{X}")

    if doc not in cache:
        cache[doc] = prompts(doc)
    system, schema = cache[doc]
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": R.format_user(question, chunks, memo, v)}]
    print(f"{B}④ 모델 판정{X} {D}({mid}, 답을 만드는 중…){X}", flush=True)
    try:
        text, ttft, total, finish, usage = R.ask(PORT, mid, messages, R.lock_version(schema, v), MAX_TOKENS, 600)
    except Exception as e:
        print(f"   오류: {e}")
        return
    try:
        o = json.loads(text)
    except json.JSONDecodeError:
        print(f"   JSON이 깨졌습니다 (종료 사유 {finish}). 원문 앞부분: {text[:300]}")
        return
    new = r3(question, o)
    state = o.get("상태")
    shown = f"{Y}{new}{X} {D}(모델은 '{state}' → R3 후처리){X}" if new else f"{Y}{state}{X}"
    print(f"   {B}판정{X}  {shown}")
    print(f"   {B}판본{X}  {o.get('적용_판본')}" + (f"  {D}기준일 {o.get('기준일')}{X}" if o.get('기준일') else ""))
    cond = o.get("조건") or {}
    if any(cond.get(k) is not None for k in ("합산액", "상한")):
        print(f"   {B}조건{X}  합산 {cond.get('합산액')}만 원 / 상한 {cond.get('상한')}만 원 · 사전 합의 {cond.get('사전_합의')}")
    gs = "; ".join(f"{g.get('출처')}{' ' + g['행'] if g.get('행') else ''}{' (' + str(g['쪽']) + '쪽)' if g.get('쪽') else ''}" for g in (o.get("근거") or []))
    print(f"   {B}근거{X}  {gs or '-'}")
    print(f"   {B}답{X}    {o.get('답')}")
    print(f"   {B}다음 행동{X} {o.get('다음_행동')}")
    if o.get("추가_확인_질문"):
        print(f"   {B}되묻기{X} {o.get('추가_확인_질문')}")
    for b in o.get("분기") or []:
        print(f"   {D}분기: {b.get('조건')} → {b.get('상태')}{X}")
    print(f"   {D}응답 {total:.1f}초 · 출력 {usage.get('completion_tokens')}토큰 · 입력 {usage.get('prompt_tokens')}토큰{X}")
    log.write(json.dumps({"시각": time.strftime("%H:%M:%S"), "모델ID": mid, "문서": doc, "질문": question, "판본_규칙": v,
                          "검색_조각": [{"출처": c["출처"], "쪽": c["쪽"]} for c in chunks], "출력_원문": text,
                          "R3_상태": new, "응답_시간_초": round(total, 2)}, ensure_ascii=False) + "\n")
    log.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4-12b-qat")
    ap.add_argument("--docs", default="V2,V3", help="판정할 문서 (쉼표). 개정 전후 비교는 V2,V3")
    ap.add_argument("--build-index", action="store_true", help="V2 · V3 벡터 색인을 만든다(처음 한 번)")
    ap.add_argument("--gpu-index", action="store_true", help="색인을 GPU로 만든다(GPU가 비어 있을 때만)")
    ap.add_argument("--ask", help="질문만 판정하고 끝낸다 (:1 · :2 · :3도 됨, 여러 개는 | 로 구분)")
    a = ap.parse_args()

    if a.build_index:
        if a.gpu_index:
            os.environ["CUDA_VISIBLE_DEVICES"] = _GPU_ENV or "0"
        build_index(["V2", "V3"], "cuda" if a.gpu_index else "cpu")
        return

    dev = {json.loads(l)["문항번호"]: json.loads(l)["질문"] for l in open(os.path.join(HERE, "data", "개발문항_150.jsonl"), encoding="utf-8")}
    PRESETS["3"] = dev["D117"]

    print("검색 준비 중 (bge-m3 불러오기)…", flush=True)
    enc = Encoder() if any((ROOT / "data" / d / "emb.npy").exists() for d in ("V2", "V3")) else None
    searchers = {d: load_searcher(d, enc) for d in ("V2", "V3")}
    for s in searchers.values():  # 첫 검색이 느리지 않게 미리 한 번 돌린다
        s.retrieve("준비", k=1, max_refs=0)
    docs = a.docs.split(",")
    mid = a.model
    if not ensure_server(mid):
        return
    cache = {}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    log = open(os.path.join(HERE, "results", "현장시연.jsonl"), "a", encoding="utf-8")

    def run(q):
        q = PRESETS.get(q.lstrip(":"), q) if re.fullmatch(r":?[123]", q) else q
        print(f"\n{B}질문{X} {q}")
        for d in docs:
            judge(q, d, searchers[d], mid, cache, log)

    if a.ask:
        for q in a.ask.split("|"):
            run(q.strip())
        return
    print(f"\n준비 완료. 모델 {mid} · 문서 {','.join(docs)}")
    print("질문을 입력하세요. 명령: :1 :2 :3 시연 질문 · :model <ID> · :docs V3 · :q 끝")
    while True:
        try:
            q = input(f"\n{B}질문>{X} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in (":q", ":quit", "exit"):
            break
        if q.startswith(":model"):
            new = q.split(maxsplit=1)[1].strip() if len(q.split()) > 1 else ""
            if new and ensure_server(new):
                mid = new
            print(f"모델: {mid}")
            continue
        if q.startswith(":docs"):
            docs = (q.split(maxsplit=1)[1] if len(q.split()) > 1 else "V2,V3").replace(" ", "").split(",")
            print(f"문서: {','.join(docs)}")
            continue
        run(q)
    print("끝. 모델 서버는 켜 둡니다 (끄기: pkill -f \"vllm serve\")")


if __name__ == "__main__":
    main()
