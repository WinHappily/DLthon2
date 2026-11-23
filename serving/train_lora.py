# [초안 · GPU에서 아직 안 돌려 봄] 학생 LoRA 학습: 입력 = 지시문 + 검색 조각 + 메모 + 판본 줄 + 질문, 정답 = 교사 JSON
# 필요: pip install peft  (transformers · torch는 팟에 있음. vLLM과 충돌하면 python -m venv --system-site-packages 로 따로)
# 사용: python train_lora.py data/sft_V3.jsonl /workspace/models/qwen3-4b-instruct-2507 /workspace/lora_qwen3-4b
# 설정: LoRA r16 · alpha32 · dropout 0.05, 대상 = 모든 선형층(q k v o gate up down), 학습률 1e-4, 2에폭, 코사인,
#       배치 1 × 누적 8, bf16, 최대 길이 8192토큰(넘는 문항은 뺌), 손실은 교사 답(assistant) 토큰에만
import json, sys
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

data_path, base, out = sys.argv[1:4]
MAXLEN = 16384
tok = AutoTokenizer.from_pretrained(base)
rows = [json.loads(l) for l in open(data_path, encoding="utf-8")]


def encode(msgs):
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = prompt + msgs[-1]["content"] + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(p_ids) + ids[len(p_ids):]
    return {"input_ids": ids, "labels": labels}


data = [e for e in (encode(r["messages"]) for r in rows) if len(e["input_ids"]) <= MAXLEN]
print(f"학습 문항 {len(data)} / {len(rows)} (길이 {MAXLEN} 넘는 문항 제외)")


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch])
    lab = torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch])
    att = torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch])
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


try:
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
model = model.to("cuda")
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()
args = TrainingArguments(output_dir=out, per_device_train_batch_size=1, gradient_accumulation_steps=8, learning_rate=1e-4,
                         num_train_epochs=2, lr_scheduler_type="cosine", warmup_steps=max(1, round(0.03 * 2 * len(data) / 8)), bf16=True, logging_steps=10,
                         save_strategy="epoch", report_to=[], remove_unused_columns=False, seed=0)
Trainer(model=model, args=args, train_dataset=data, data_collator=collate).train()
model.save_pretrained(out)
tok.save_pretrained(out)
print("저장:", out)
