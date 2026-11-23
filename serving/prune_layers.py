# 층 빼기 가지치기 (ShortGPT 방식, 재학습 없음): 층마다 '들어온 값과 나간 값이 얼마나 같은지'를 재서
# 가장 덜 바꾸는 층부터 뺀다(첫 층·마지막 층은 남김). 파일 크기·메모리가 실제로 줄어드는 구조적 가지치기
# 실행: python prune_layers.py <원본 폴더> <저장 앞머리> <뺄 층 수들, 예 4,8> <보정 문장 jsonl>
#   → <저장 앞머리>4, <저장 앞머리>8 폴더 (vLLM으로 그대로 띄움). 층 점수는 <저장 앞머리>_층점수.json
import json, sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

src, dst, ns, calib = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")], sys.argv[4]
tok = AutoTokenizer.from_pretrained(src)


def load():  # transformers 판에 따라 dtype / torch_dtype
    try:
        return AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16)


model = load().to("cuda")
model.eval()
layers = model.model.layers
L = len(layers)
sim = torch.zeros(L, dtype=torch.float64)
cnt = torch.zeros(L, dtype=torch.float64)


def hook(i):
    def f(mod, args, kwargs, out):
        h_in = args[0] if args else kwargs["hidden_states"]
        h_out = out[0] if isinstance(out, (tuple, list)) else out
        sim[i] += F.cosine_similarity(h_in.float(), h_out.float(), dim=-1).mean().item()
        cnt[i] += 1
    return f


hs = [layers[i].register_forward_hook(hook(i), with_kwargs=True) for i in range(L)]
texts = [json.loads(l)["text"] for l in open(calib, encoding="utf-8")]
with torch.no_grad():
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(model.device)
        model(ids)
for h in hs:
    h.remove()
score = (1 - sim / cnt.clamp(min=1)).tolist()  # 클수록 많이 바꾸는(중요한) 층
order = sorted(range(1, L - 1), key=lambda i: score[i])  # 첫·마지막 층 제외, 덜 중요한 순
json.dump({"층수": L, "중요도": score, "빼는_순서": order}, open(f"{dst}_층점수.json", "w"), ensure_ascii=False, indent=1)
print("층 중요도(낮은 순 8개):", [(i, round(score[i], 4)) for i in order[:8]])
del model
torch.cuda.empty_cache()

for n in ns:
    m = load()  # CPU에서 다시 읽어 층을 뺀다
    drop = set(order[:n])
    keep = [i for i in range(L) if i not in drop]
    m.model.layers = torch.nn.ModuleList([m.model.layers[i] for i in keep])
    for j, layer in enumerate(m.model.layers):  # 캐시 번호를 새 순서로
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = j
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = j
    c = m.config
    c.num_hidden_layers = len(keep)
    if getattr(c, "layer_types", None):
        c.layer_types = [c.layer_types[i] for i in keep]
    if getattr(c, "max_window_layers", None):
        c.max_window_layers = min(c.max_window_layers, len(keep))
    m.save_pretrained(f"{dst}{n}", safe_serialization=True)
    tok.save_pretrained(f"{dst}{n}")
    print(f"층 {L} → {len(keep)} (뺀 층 {sorted(drop)}) 저장: {dst}{n}")
    del m
