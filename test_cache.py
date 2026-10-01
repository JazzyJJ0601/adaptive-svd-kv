#!/usr/bin/env python3
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_PATH = "/home/jasper/eirene-projects/03-inference-lab/ai-lab/models/Qwen--Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, local_files_only=True, dtype=torch.bfloat16,
    device_map="auto", attn_implementation="sdpa"
)
model.eval()

# Run inference
inputs = tokenizer("test input", return_tensors="pt")
outputs = model(**inputs, use_cache=True)

cache = outputs.past_key_values
print(f"Type: {type(cache)}")
print(f"Has layers attr: {hasattr(cache, 'layers')}")
if hasattr(cache, 'layers'):
    print(f"Number of layers: {len(cache.layers)}")
    print(f"First layer type: {type(cache.layers[0])}")
