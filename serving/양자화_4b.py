#!/usr/bin/env python3
"""증류한 4B(LoRA를 합친 16비트 모델)를 GPTQ 4비트(W4A16)로 줄인다. vLLM이 바로 읽는 형식(compressed-tensors)으로 저장한다.

보정 데이터 = 증류 학습 대화(sft_*.jsonl) 128개. 학생이 실제로 받는 입력(지시문 + 검색 조각 + 메모 + 판본)과 교사 답 그대로라
4비트로 줄일 때 오차를 실제 쓰임새에 맞춰 줄인다. 평가 문항은 쓰지 않는다.

실행(llmcompressor는 vLLM과 따로 둔 가상환경에서):
  source /workspace/venv_lora/bin/activate && pip install llmcompressor
  python 양자화_4b.py [학습대화.jsonl]
결과: /workspace/models/qwen3-4b-lora-w4a16
"""
import glob
import json
import os
import random
import sys
import time

import torch
import transformers
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = "/workspace/models/qwen3-4b-lora"
OUT = "/workspace/models/qwen3-4b-lora-w4a16"
N = int(os.environ.get("CALIB_N", 128))          # 보정 대화 수 (빨리 해 볼 때 32)
MAXLEN = int(os.environ.get("CALIB_LEN", 8192))  # 대화당 최대 토큰 (빨리 해 볼 때 4096)

sft = sys.argv[1] if len(sys.argv) > 1 else None
if not sft:
    found = sorted(glob.glob("/workspace/**/sft_*V3*.jsonl", recursive=True) + glob.glob("/workspace/**/sft_*.jsonl", recursive=True))
    if not found:
        sys.exit("학습 대화 파일(sft_*.jsonl)을 못 찾았습니다. 경로를 인자로 주세요")
    sft = found[0]
print("보정 데이터:", sft)

tok = AutoTokenizer.from_pretrained(SRC)
rows = [json.loads(l) for l in open(sft, encoding="utf-8") if l.strip()]
random.Random(0).shuffle(rows)
enc = [tok.apply_chat_template(r["messages"], tokenize=False) for r in rows[:N]]
ds = Dataset.from_list([tok(t, truncation=True, max_length=MAXLEN, add_special_tokens=False) for t in enc])
print(f"보정 {len(ds)}개, 평균 {sum(len(x) for x in ds['input_ids']) / len(ds):.0f}토큰")

kw = {"dtype": torch.bfloat16} if int(transformers.__version__.split(".")[0]) >= 5 else {"torch_dtype": torch.bfloat16}
model = AutoModelForCausalLM.from_pretrained(SRC, device_map="auto", **kw)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

t0 = time.time()
oneshot(model=model, dataset=ds, recipe=GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
        max_seq_length=MAXLEN, num_calibration_samples=len(ds))
model.save_pretrained(OUT, save_compressed=True)
tok.save_pretrained(OUT)
size = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.endswith(".safetensors")) / 1e9
print(f"끝: {OUT} ({size:.2f} GB, {time.time() - t0:.0f}초)")
