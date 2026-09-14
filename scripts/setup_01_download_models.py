# -*- coding: utf-8 -*-
"""从 ModelScope 下载本地模型到 rag_qa/models 与 rag_qa/core/bert_query_classifier"""
import os
from modelscope import snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "rag_qa", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# (modelscope 模型ID, 本地目标目录)
TARGETS = [
    ("BAAI/bge-m3", os.path.join(MODELS_DIR, "bge-m3")),
    ("BAAI/bge-reranker-large", os.path.join(MODELS_DIR, "bge-reranker-large")),
    # 意图分类基座（微调产物将保存到 rag_qa/core/bert_query_classifier）
    ("AI-ModelScope/bert-base-chinese",
     os.path.join(MODELS_DIR, "bert-base-chinese")),
]

for model_id, dest in TARGETS:
    if os.path.exists(dest) and any(f.endswith((".bin", ".safetensors")) for f in os.listdir(dest)):
        print(f"=== 已存在，跳过: {dest} ===", flush=True)
        continue
    print(f"=== 下载 {model_id} -> {dest} ===", flush=True)
    path = snapshot_download(model_id, local_dir=dest)
    print(f"完成: {path}", flush=True)

print("全部模型下载完成")
