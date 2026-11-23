"""소형 후보 모델을 /workspace/models/<이름>에 내려받고 검사한다.

실행 (Pod 터미널):
  cd /workspace/DLthon02/ver2
  nohup python -u download_models.py > download.log 2>&1 &   # -u: 출력을 바로 로그에 쓴다
  tail -f download.log          # 진행 보기. Ctrl+C는 보기만 멈추고 다운로드는 계속된다
중간에 끊겨도 다시 실행하면 받은 파일은 건너뛰고 이어 받는다.
비공개(gated) 저장소는 미리 `export HF_TOKEN=...`을 터미널에서 해 둔다(토큰을 채팅에 붙이지 않는다).
"""
import fnmatch, json, os, shutil, sys, time

from huggingface_hub import HfApi, snapshot_download

MODELS_DIR = "/workspace/models"
VOLUME_GB = 100  # RunPod 볼륨 크기. df가 실제 할당량을 못 보여 주는 경우를 대비해 직접 계산에도 쓴다
MARGIN_GB = 5    # 받은 뒤에도 남겨 둘 여유

# (run_all에서 쓸 이름 = 폴더 이름, 허깅페이스 저장소). 받을 것만 주석을 푼다.
TODO = [
    ("qwen3-4b-instruct-2507", "Qwen/Qwen3-4B-Instruct-2507"),
    ("kanana2-3b", "kakaocorp/kanana-2-3b-instruct"),
    ("qwen3.5-4b", "Qwen/Qwen3.5-4B"),
    ("gemma4-e4b", "google/gemma-4-E4B-it"),
]
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py", "LICENSE*", "README*"]
GB = 1024 ** 3


def used_bytes(root):
    total = 0
    for d, _, files in os.walk(root):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(d, f))
            except OSError:
                pass
    return total


def free_bytes():
    by_df = shutil.disk_usage("/workspace").free
    by_quota = VOLUME_GB * GB - used_bytes("/workspace")
    return min(by_df, by_quota)


def remote_files(repo):
    info = HfApi().model_info(repo, files_metadata=True)
    return {s.rfilename: (s.size or 0) for s in info.siblings
            if any(fnmatch.fnmatch(os.path.basename(s.rfilename), p) for p in PATTERNS)}


def check(name, local, files):
    """받은 폴더를 검사하고 run_all에 넣을 줄을 출력한다."""
    ok = True
    for f, size in files.items():
        p = os.path.join(local, f)
        if not os.path.exists(p):
            print(f"  [누락] {f}"); ok = False
        elif f.endswith(".safetensors") and size and os.path.getsize(p) != size:
            print(f"  [크기 불일치] {f}: {os.path.getsize(p)} != {size}"); ok = False
    cfg = json.load(open(os.path.join(local, "config.json"), encoding="utf-8"))
    tc = os.path.join(local, "tokenizer_config.json")
    has_tmpl = os.path.exists(os.path.join(local, "chat_template.jinja")) or \
        (os.path.exists(tc) and "chat_template" in json.load(open(tc, encoding="utf-8")))
    print(f"  구조 {cfg.get('architectures')}, dtype {cfg.get('torch_dtype') or cfg.get('dtype')}, "
          f"최대 문맥 {cfg.get('max_position_embeddings')}, 채팅 템플릿 {'있음' if has_tmpl else '없음(확인 필요)'}")
    print(f"  폴더 크기 {used_bytes(local) / GB:.1f} GB")
    print(f"  {'검사 통과' if ok else '검사 실패: 다시 실행해 이어 받기'}")
    print(f"  run_all 모델ID: {name}")
    return ok


for name, repo in TODO:
    local = os.path.join(MODELS_DIR, name)
    print(f"\n=== {name} ← {repo}")
    try:
        files = remote_files(repo)
    except Exception as e:
        print(f"  저장소 정보를 못 가져옴: {type(e).__name__}: {e}")
        continue
    have = sum(os.path.getsize(os.path.join(local, f)) for f in files if os.path.exists(os.path.join(local, f)))
    need = sum(files.values()) - have
    free = free_bytes()
    print(f"  받을 양 {need / GB:.1f} GB (전체 {sum(files.values()) / GB:.1f} GB), 남은 공간 {free / GB:.1f} GB")
    if need > free - MARGIN_GB * GB:
        print(f"  [건너뜀] 공간 부족. 여유 {MARGIN_GB} GB를 남기려면 {(need - free) / GB + MARGIN_GB:.1f} GB를 더 비워야 한다")
        continue
    for attempt in range(1, 4):
        try:
            t0 = time.time()
            snapshot_download(repo_id=repo, local_dir=local, allow_patterns=PATTERNS, max_workers=8)
            print(f"  다운로드 완료 {time.time() - t0:.0f}초")
            break
        except Exception as e:
            print(f"  [시도 {attempt}/3 실패] {type(e).__name__}: {e}")
            time.sleep(10)
    else:
        print("  세 번 모두 실패. 네트워크나 저장소 이름을 확인한다")
        continue
    check(name, local, files)
