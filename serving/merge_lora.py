# [초안] LoRA를 원본에 합쳐 vLLM이 그대로 읽는 모델 폴더를 만든다 → 평가_실행.sh qwen3-4b-lora - v3p3
# 사용: python merge_lora.py /workspace/models/qwen3-4b-instruct-2507 /workspace/lora_qwen3-4b /workspace/models/qwen3-4b-lora
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base, lora, out = sys.argv[1:4]
try:
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
except TypeError:
    m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, lora).merge_and_unload()
m.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
print("저장:", out)
