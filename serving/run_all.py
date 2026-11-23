#!/usr/bin/env python3
"""DLthon2 모델 비교 실행기.

모델마다 vLLM 서버를 띄우고, 문항을 하나씩 보내 답을 받은 뒤 서버를 끄고 다음 모델로 넘어간다.
결과는 문항마다 한 줄씩 JSONL로 바로 저장한다. 이미 오류 없이 저장된 (실험ID, 모델ID, 문항번호)는 건너뛴다.

예) T1, 검색 없이 전 모델:
  python run_all.py --questions 질문.jsonl --examples 예시.json --exp T1 --out results_T1.jsonl
예) 시험 삼아 한 모델 3문항:
  python run_all.py --questions 질문.jsonl --examples 예시.json --exp TEST --out test.jsonl \
      --models qwen3-4b-awq --limit 3
"""
import argparse
import csv
import json
import os
import re
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

MODELS_DIR = "/workspace/models"
PROMPT_VERSION = "p2"  # 아래 SYSTEM_PROMPT를 고치면 올린다

# 실행 순서대로. 값은 추가 vLLM 인자 후보들이며, 앞 후보로 서버가 안 뜨면 다음 후보로 다시 띄운다.
MODELS = {
    "qwen3-32b-awq": [[]],
    "qwen3-4b-awq": [[]],
    "exaone3.5-2.4b-awq": [["--trust-remote-code"]],
    "exaone3.5-32b-awq": [["--trust-remote-code"]],
    "kanana2-3b": [["--trust-remote-code"]],
    "gemma4-31b-qat": [[]],
    "qwen3.8-27b-awq": [["--language-model-only"], []],
    # 11/19 소형 A군 (kanana2-3b는 위에 있음). 비전·오디오 인코더를 빼고 띄우고, 안 되면 그대로 띄운다
    "qwen3-4b-instruct-2507": [[]],
    "qwen3.5-4b": [["--language-model-only"], []],
    "gemma4-e4b": [["--language-model-only"], []],
    # 11/19 밤: 32B급 8비트(기준선 참고). A40은 FP8 연산이 없어 FP8 Marlin(가중치만 8비트)으로 돈다. 안 뜨면 문맥을 줄여 다시
    "qwen3.8-27b-fp8": [["--language-model-only"], ["--language-model-only", "--max-model-len", "24576", "--gpu-memory-utilization", "0.95"]],
    "gemma4-31b-fp8": [["--language-model-only"], ["--language-model-only", "--max-model-len", "24576", "--gpu-memory-utilization", "0.95"]],
    # 4비트 7~14B (공식 양자화본)
    "gemma4-12b-qat": [["--language-model-only"], []],
    "qwen3-14b-awq": [[]],
    "qwen3-8b-awq": [[]],
    "exaone3.5-7.8b-awq": [["--trust-remote-code"]],
    # 1.5B급 (16비트 원본). kanana 1.3B는 독자 구조라 안 뜨면 transformers 백엔드로
    "kanana2-1.3b": [["--trust-remote-code"], ["--trust-remote-code", "--model-impl", "transformers"]],
    "qwen3.5-2b": [["--language-model-only"], []],
    "exaone4.0-1.2b": [[], ["--trust-remote-code"]],
    "gemma4-e2b": [["--language-model-only"], []],
    # 11/19 밤: 32B급 8비트(FP8, A40에선 가중치만 8비트), 4비트 7~14B, 1.5B급
    "qwen3-32b-fp8": [[], ["--max-model-len", "24576", "--gpu-memory-utilization", "0.95"]],
    # 11/20 주말: qwen3-4b-instruct-2507에서 층을 뺀 것(prune_layers.py)
    "qwen3-4b-prune4": [[]],
    "qwen3-4b-prune8": [[]],
    # 증류 결과(merge_lora.py로 합친 폴더 /workspace/models/qwen3-4b-lora)
    "qwen3-4b-lora": [[]],
    # 11/23: 증류 4B를 GPTQ 4비트로 줄인 것(양자화_4b.py)
    "qwen3-4b-lora-w4a16": [[]],
}

# 전 모델 공통 서버 설정. 응답 시간·메모리 비교를 위해 같게 둔다.
COMMON_ARGS = [
    "--max-model-len", "32768",  # p2: 지시문이 길어지고 T2+메모는 1.5만 자를 넘을 수 있어 16384에서 올림
    "--gpu-memory-utilization", "0.90",
    "--max-num-seqs", "8",
    "--seed", "0",
    "--generation-config", "vllm",  # 모델별 기본 샘플링 설정(반복 패널티 등)을 쓰지 않음
]

SYSTEM_PROMPT = """너는 신혼생활 운영위원회 공동생활 규약집(V2 합본)을 근거로 사용자의 상황을 판정한다.
주어진 규약 조각만 근거로 쓰고, 조각에 없는 수치나 조항은 지어내지 않는다.

판정 절차
1. 질문_범위를 고른다. 규약이 다루는 생활 문제면 "규약 질문", 잡담·메뉴 고민처럼 규약과 상관없으면 "규약과 무관", 뜻을 알아볼 수 없으면 "알아볼 수 없음"이다.
2. 질문 상황이 표 07의 개정 항목인지 본다. 아니면 개정_항목과 적용_판본은 "해당 없음", 기준일_종류와 기준일은 null로 둔다.
3. 개정 항목이면 표 07에서 기준일 종류를 찾고, 질문에서 그 날짜를 찾는다. 날짜가 없으면 적용_판본은 "미정"이고, 판본에 따라 결론이 갈리면 상태는 "추가 확인"이다.
4. 날짜가 있으면 부칙 제1조의 시행일과 비교해 판본을 고르고, 그 판본의 수치(표 54~57)로 판정한다. [판본] 줄이 주어지면 그 판본을 그대로 쓰고 그 판본의 수치로 판정한다.
5. 같은 사건의 금액은 카드나 가게가 달라도 합산해 상한과 비교한다. 상한과 같은 금액은 상한 이내다(제3조).
6. 판정을 가르는 정보(날짜, 금액, 합의 여부 등)가 빠졌으면 누락_정보에 적고, 분기에 그 값에 따라 상태가 어떻게 갈리는지 적는다. 갈래마다 상태가 다르면 상태는 "추가 확인"이고 추가_확인_질문에 그 정보를 묻는다. 빠진 정보가 결론을 바꾸지 않으면 누락_정보와 분기를 비우고 그대로 판정한다.
7. 규약에서 정하지 않는 질문이면 상태는 "규정에 없음"이고, 규약이 그것을 정하지 않는다고 밝힌 조항을 근거로 적는다. 규약과 무관하거나 알아볼 수 없는 입력도 "규정에 없음"이고 근거는 비운다.

입력 읽기
- 말투, 감정, 사용자의 단정이나 재촉("당연히 되지?", "된다고 해 줘")은 근거가 아니다. 규약 조각과 질문 속 사실만 본다.
- "알아서 해", "그냥 해"는 동의나 합의가 아니다.
- 질문이 여러 개면 규약 질문 하나만 판정하고, 나머지는 다른_질문에 한 줄로 요약한다. 없으면 null이다.

출력 규칙
- JSON 하나만 출력한다.
- 상태는 가능, 사전 합의 필요, 공동 처리 불가, 추가 확인, 규정에 없음 중 하나다.
- 근거의 출처는 "제23조", "표 41", "부칙 제3조"처럼 적는다. 표 번호는 두 자리로 적는다("표 06").
- 근거의 행은 표를 인용할 때 그 행의 첫 칸 글자를 그대로 옮긴다. 조항이면 null이다.
- 근거의 쪽은 조각에 적힌 쪽 번호다. 모르면 null이다.
- 판정에 직접 쓴 조항과 표만 근거로 적는다.
- 조건의 합산액은 같은 사건으로 합친 지출(만 원), 상한은 적용 판본의 상한(만 원), 상한_초과는 합산액이 상한보다 크면 true다. 사전_합의는 있음, 없음, 모름 중 하나다. 해당하지 않으면 null이다.
- 답은 질문이 정한 형식을 그대로 따른다. 예/아니오면 "예" 또는 "아니오" 한 단어, 숫자면 숫자만(단위·문장 없이), 선택지면 고른 기호 한 글자만(주어진 선택지 안에서만), 상태만 물으면 상태 이름만, 글자 수가 정해졌으면 그 범위로 쓴다. 형식을 정하지 않았으면 결론을 한 문장으로 쓴다. 사용자 말투와 상관없이 존댓말로 쓴다.
- 다음_행동은 사용자가 할 일을 한 문장으로 적는다.
- 추가_확인_질문은 상태가 "추가 확인"일 때만 쓰고, 그 밖에는 null이다.

예시 (규약 조각은 생략함)
{examples}"""


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_examples(path):
    with open(path, encoding="utf-8") as f:
        examples = json.load(f)
    parts = []
    for i, ex in enumerate(examples, 1):
        out = ex["출력"] if isinstance(ex["출력"], str) else json.dumps(ex["출력"], ensure_ascii=False)
        parts.append(f"예시 {i} ({ex['상태']})\n질문: {ex['질문']}\n출력: {out}")
    return "\n\n".join(parts)


def build_schema(opt):
    nullable = lambda t: {"type": [t, "null"]}
    props = {  # 칸 순서 = 생성 순서. 판본과 분기를 상태보다 먼저 쓰게 한다
        "질문_범위": {"enum": ["규약 질문", "규약과 무관", "알아볼 수 없음"]},
        "다른_질문": nullable("string"),
        "개정_항목": {"enum": opt["개정_항목"]},
        "기준일_종류": {"anyOf": [{"enum": opt["기준일_종류"]}, {"type": "null"}]},
        "기준일": nullable("string"),
        "적용_판본": {"enum": opt["적용_판본"]},
        "조건": {  # p2 고정 키. 기준일은 위 칸을 쓴다
            "type": "object",
            "properties": {"합산액": nullable("number"), "상한": nullable("number"), "상한_초과": nullable("boolean"),
                           "사전_합의": {"anyOf": [{"enum": ["있음", "없음", "모름"]}, {"type": "null"}]}},
            "required": ["합산액", "상한", "상한_초과", "사전_합의"],
            "additionalProperties": False,
        },
        "누락_정보": {"type": "array", "items": {"type": "string"}},
        "분기": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"조건": {"type": "string"}, "상태": {"enum": opt["상태"]}},
                "required": ["조건", "상태"],
                "additionalProperties": False,
            },
        },
        "근거": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"출처": {"type": "string"}, "행": nullable("string"), "쪽": nullable("integer")},
                "required": ["출처", "행", "쪽"],
                "additionalProperties": False,
            },
        },
        "상태": {"enum": opt["상태"]},
        "답": {"type": "string"},
        "다음_행동": {"type": "string"},
        "추가_확인_질문": nullable("string"),
    }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def version_line(v):
    """A 규칙 판본(make_version.py 한 줄)을 모델에게 알리는 줄"""
    if v.get("규칙_표기") == "규정 범위 밖":
        return '[판본] 사건일이 최초 시행일(표 53) 전이다. 적용_판본은 "해당 없음"이다.'
    if v["판본"] == "해당 없음":
        return '[판본] 이 질문은 개정 항목이 아니다. 적용_판본은 "해당 없음"이다.'
    if v["판본"] == "미정":
        return f'[판본] 개정 항목({v["항목"]})인데 판본을 가를 기준일이 질문에 없다. 적용_판본은 "미정"이다.'
    return f'[판본] 적용_판본은 {v["판본"]}이다({v["항목"]}, 기준일 {v["기준일"]}). 이 판본의 수치로 판정한다.'


def format_user(question, chunks, memo=None, version=None):
    if chunks:
        body = "\n\n".join(f"<{c['출처']} | {c.get('쪽')}쪽>\n{c['본문']}" for c in chunks)
    else:
        body = "(없음)"
    parts = [f"[규약 조각]\n{body}"]
    if memo:  # A 판정 메모(T2 + 메모 조건)
        parts.append(memo)
    if version:  # A 규칙 판본(판본 고정 조건)
        parts.append(version_line(version))
    parts.append(f"[질문]\n{question}")
    return "\n\n".join(parts)


def lock_version(schema, v):
    """문항마다 적용_판본 칸을 규칙 값 하나로 묶은 형식을 만든다"""
    s = json.loads(json.dumps(schema))
    s["properties"]["적용_판본"] = {"enum": [v["판본"]]}
    return s


def load_retrieval(path, pages_path):
    """검색 결과(가정): 한 줄에 {"문항번호": ..., "조각": [{"출처", "쪽", "본문", "경로"}, ...]}"""
    rows = {r["문항번호"]: r["조각"] for r in load_jsonl(path)}
    page_of = {}
    if pages_path:
        with open(pages_path, encoding="utf-8-sig") as f:
            page_of = {r["출처"]: int(r["쪽"]) for r in csv.DictReader(f)}
    for qid, chunks in rows.items():
        for c in chunks:
            m = re.fullmatch(r"표 (\d+)", c["출처"])
            expected = int(m.group(1)) if m else page_of.get(c["출처"])
            if expected is not None and c.get("쪽") != expected:
                print(f"[경고] {qid} 조각 {c['출처']}의 쪽 {c.get('쪽')} != 규약집 {expected}")
    return rows


def server_alive(port):
    try:
        return requests.get(f"http://localhost:{port}/health", timeout=5).status_code == 200
    except requests.RequestException:
        return False


def wait_gpu_free(limit_s=120):
    t0 = time.time()
    while time.time() - t0 < limit_s:
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip()
        if used and int(used.splitlines()[0]) < 2000:
            return
        time.sleep(3)


def stop_server(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    wait_gpu_free()


def start_server(mid, port, log_dir, timeout_s):
    log_path = os.path.join(log_dir, f"{mid}.log")
    for extra in MODELS[mid]:
        cmd = ["vllm", "serve", os.path.join(MODELS_DIR, mid), "--served-model-name", mid,
               "--port", str(port), *COMMON_ARGS, *extra]
        log = open(log_path, "a", encoding="utf-8")
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        offset = os.path.getsize(log_path)
        t0 = time.time()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while time.time() - t0 < timeout_s and proc.poll() is None:
            if server_alive(port):
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    f.seek(offset)
                    found = re.findall(r"Model loading took ([\d.]+) ?GiB", f.read())
                return proc, {"서버_인자": extra, "서버_시작_초": round(time.time() - t0, 1),
                              "가중치_메모리_GiB": float(found[-1]) if found else None}
            time.sleep(5)
        stop_server(proc)
        print(f"[{mid}] 서버 시작 실패 (추가 인자 {extra or '없음'}). 로그: {log_path}")
    return None, {"서버_인자": None, "서버_시작_초": None, "가중치_메모리_GiB": None}


def ask(port, mid, messages, schema, max_tokens, timeout_s):
    body = {
        "model": mid, "messages": messages, "stream": True, "stream_options": {"include_usage": True},
        "temperature": 0, "seed": 0, "max_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": {"name": "판정", "schema": schema, "strict": True}},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    ttft, pieces, finish, usage = None, [], None, {}
    with requests.post(f"http://localhost:{port}/v1/chat/completions", json=body, stream=True, timeout=timeout_s) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        for raw in r.iter_lines():
            line = raw.decode("utf-8")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage") or usage
            for ch in chunk.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    pieces.append(piece)
                finish = ch.get("finish_reason") or finish
    total = time.perf_counter() - t0
    return "".join(pieces), ttft, total, finish, usage


def dir_size_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            p = os.path.join(root, name)
            if not os.path.islink(p):
                total += os.path.getsize(p)
    return round(total / 1e9, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--examples", required=True)
    ap.add_argument("--options", default="options_v2.json")
    ap.add_argument("--retrieval", help="검색 결과 JSONL. 없으면 조각 없이 보낸다(T1)")
    ap.add_argument("--retrieval-ver", default="없음")
    ap.add_argument("--pages", help="조항 쪽번호표 CSV. 검색 조각의 쪽 검사용")
    ap.add_argument("--exp", required=True, help="실험ID (예: T1, T2)")
    ap.add_argument("--qset", default="q20")
    ap.add_argument("--rule-ver", default="없음")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", help="쉼표로 구분한 모델ID. 기본은 전부")
    ap.add_argument("--limit", type=int, help="앞에서부터 N문항만")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--use-memo", action="store_true", help="검색 결과의 '메모'(A 판정 메모)를 붙인다")
    ap.add_argument("--version-file", help="A 규칙 판본 파일(make_version.py 출력). 주면 적용_판본을 고정")
    ap.add_argument("--concurrency", type=int, default=1, help="동시 요청 수. 1이 아니면 응답 시간은 지표로 쓰지 않는다")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--startup-timeout", type=int, default=1200)
    ap.add_argument("--request-timeout", type=int, default=300)
    ap.add_argument("--system-file", help="지시문 파일. 없으면 SYSTEM_PROMPT(p2). 파일 이름이 프롬프트_버전")
    args = ap.parse_args()

    if args.retrieval and args.retrieval_ver == "없음":
        ap.error("--retrieval을 주면 --retrieval-ver도 적어 주세요 (예: r1)")
    if server_alive(args.port):
        ap.error(f"포트 {args.port}에 이미 서버가 떠 있습니다. 이전 vLLM을 먼저 끄세요")

    questions = load_jsonl(args.questions)[: args.limit]
    with open(args.options, encoding="utf-8") as f:
        schema = build_schema(json.load(f))
    template = open(args.system_file, encoding="utf-8").read() if args.system_file else SYSTEM_PROMPT
    prompt_ver = os.path.splitext(os.path.basename(args.system_file))[0] if args.system_file else PROMPT_VERSION
    system = template.format(examples=format_examples(args.examples))
    retrieval = load_retrieval(args.retrieval, args.pages) if args.retrieval else {}
    memos = {r["문항번호"]: r.get("메모") for r in load_jsonl(args.retrieval)} if (args.retrieval and args.use_memo) else {}
    if args.use_memo and not any(memos.values()):
        ap.error("--use-memo인데 검색 결과에 '메모' 칸이 없습니다")
    versions = {r["문항번호"]: r for r in load_jsonl(args.version_file)} if args.version_file else {}
    if args.version_file:
        missing = [q["문항번호"] for q in questions if q["문항번호"] not in versions]
        bad = {v["판본"] for v in versions.values()} - set(schema["properties"]["적용_판본"]["enum"])
        if missing or bad:
            ap.error(f"판본 파일 확인: 없는 문항 {missing[:5]}, 형식에 없는 값 {bad}")
    models = args.models.split(",") if args.models else list(MODELS)
    settings = {"temperature": 0, "추론_모드": False, "최대_출력_토큰": args.max_tokens, "형식_강제": True,
                "최대_컨텍스트": int(COMMON_ARGS[1]), "판정_메모": args.use_memo,
                "판본_고정": os.path.basename(args.version_file) if args.version_file else None,
                "동시_요청": args.concurrency}

    done = set()
    if os.path.exists(args.out):
        done = {(r["실험ID"], r["모델ID"], r["문항번호"]) for r in load_jsonl(args.out) if r["오류"] is None}
    log_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    summary_path = re.sub(r"\.jsonl$", "", args.out) + "_models.jsonl"

    with open(args.out, "a", encoding="utf-8") as out:
        for mid in models:
            todo = [q for q in questions if (args.exp, mid, q["문항번호"]) not in done]
            if not todo:
                print(f"[{mid}] 이미 끝남")
                continue
            print(f"[{mid}] 서버 시작 ({len(todo)}문항)")
            proc, info = start_server(mid, args.port, log_dir, args.startup_timeout)
            with open(summary_path, "a", encoding="utf-8") as s:
                s.write(json.dumps({"실험ID": args.exp, "모델ID": mid, "파일_크기_GB": dir_size_gb(os.path.join(MODELS_DIR, mid)),
                                    **info, "오류": None if proc else "서버 시작 실패"}, ensure_ascii=False) + "\n")
            def one(q):  # 문항 하나를 보내고 결과 행을 만든다. 동시 요청이면 여러 스레드에서 불린다
                chunks = retrieval.get(q["문항번호"], [])
                row = {
                    "문항번호": q["문항번호"], "실험ID": args.exp, "모델ID": mid,
                    "질문세트_버전": args.qset, "검색_버전": args.retrieval_ver,
                    "규칙_버전": args.rule_ver, "프롬프트_버전": prompt_ver,
                    "검색_조각": [{"출처": c["출처"], "쪽": c.get("쪽"), "경로": c.get("경로")} for c in chunks],
                    "설정": settings, "출력_원문": None, "종료_사유": None, "첫토큰_시간_초": None,
                    "응답_시간_초": None, "입력_토큰": None, "출력_토큰": None, "오류": None,
                }
                if proc is None:
                    row["오류"] = "서버 시작 실패"
                    return row
                v = versions.get(q["문항번호"])
                messages = [{"role": "system", "content": system},
                            {"role": "user", "content": format_user(q["질문"], chunks, memos.get(q["문항번호"]), v)}]
                try:
                    text, ttft, total, finish, usage = ask(args.port, mid, messages, lock_version(schema, v) if v else schema,
                                                           args.max_tokens, args.request_timeout)
                    row.update({"출력_원문": text, "종료_사유": finish,
                                "첫토큰_시간_초": round(ttft, 3) if ttft is not None else None,
                                "응답_시간_초": round(total, 3),
                                "입력_토큰": usage.get("prompt_tokens"), "출력_토큰": usage.get("completion_tokens")})
                    print(f"[{mid}] {q['문항번호']} {total:.1f}초 {finish} 출력 {usage.get('completion_tokens')}토큰")
                except Exception as e:  # 한 문항 실패가 전체를 멈추지 않게 기록만 하고 넘어간다
                    row["오류"] = f"{type(e).__name__}: {e}"
                    print(f"[{mid}] {q['문항번호']} 오류: {row['오류']}")
                return row

            pool = ThreadPoolExecutor(max_workers=args.concurrency)
            try:
                for f in as_completed([pool.submit(one, q) for q in todo]):
                    out.write(json.dumps(f.result(), ensure_ascii=False) + "\n")
                    out.flush()
            finally:  # 셀을 중간에 멈춰도 남은 요청은 취소하고 서버는 끈다
                pool.shutdown(wait=False, cancel_futures=True)
                if proc is not None:
                    stop_server(proc)


if __name__ == "__main__":
    main()
