# -*- coding: utf-8 -*-
"""训练意图二分类 BERT，并保存到 rag_qa/core/bert_query_classifier

注意：始终从独立的基座目录 rag_qa/models/bert-base-chinese 加载，
微调产物保存到另一目录，避免 Windows 下覆盖被自身 mmap 锁定的权重文件。
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rag_qa.core.query_classifier import QueryClassifier

BASE_DIR = os.path.join(ROOT, "rag_qa", "models", "bert-base-chinese")
TARGET_DIR = os.path.join(ROOT, "rag_qa", "core", "bert_query_classifier")
DATA_FILE = os.path.join(ROOT, "rag_qa", "classify_data", "training_data_1000.json")

# 从独立基座初始化（不加载目标目录，保存时才不会与自身 mmap 冲突）
clf = QueryClassifier(model_path=BASE_DIR)
# 训练（内部还会另存一份到 cwd/bert_outputs）
clf.train_model(data_file=DATA_FILE)

# 清空目标目录后保存微调产物
shutil.rmtree(TARGET_DIR, ignore_errors=True)
os.makedirs(TARGET_DIR, exist_ok=True)
clf.model.save_pretrained(TARGET_DIR)
clf.tokenizer.save_pretrained(TARGET_DIR)
print(f"意图模型已保存到 {TARGET_DIR}")
