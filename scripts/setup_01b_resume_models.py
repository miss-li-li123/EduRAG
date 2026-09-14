# -*- coding: utf-8 -*-
import os
import shutil
import time
import traceback
from modelscope import snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "rag_qa", "models")

bge = os.path.join(MODELS_DIR, "bge-m3")
shutil.rmtree(os.path.join(bge, "._____temp"), ignore_errors=True)

JOBS = [
    ("BAAI/bge-m3", bge,
     ["config.json", "configuration.json", "*.json", "*.txt", "*.model",
      "pytorch_model.bin", "*.pt", "1_Pooling/*", "tokenizer*",
      "special_tokens_map.json", "modules.json", "sentence_bert_config.json"]),
    ("BAAI/bge-reranker-large", os.path.join(MODELS_DIR, "bge-reranker-large"),
     ["config.json", "*.json", "*.txt", "*.model", "model.safetensors",
      "pytorch_model.bin", "*.bin", "1_Pooling/*", "tokenizer*",
      "special_tokens_map.json", "modules.json", "sentence_bert_config.json"]),
    ("AI-ModelScope/bert-base-chinese",
     os.path.join(MODELS_DIR, "bert-base-chinese"),
     ["config.json", "*.json", "*.txt", "model.safetensors",
      "pytorch_model.bin", "vocab.txt", "tokenizer*", "special_tokens_map.json"]),
]

for model_id, dest, patterns in JOBS:
    for attempt in range(1, 5):
        try:
            print(f"=== [{attempt}] {model_id} -> {dest}", flush=True)
            snapshot_download(model_id, local_dir=dest, allow_patterns=patterns)
            print(f"=== 完成 {model_id}", flush=True)
            break
        except Exception:
            traceback.print_exc()
            time.sleep(5)
    else:
        raise SystemExit(f"下载失败: {model_id}")
print("ALL_DONE")
