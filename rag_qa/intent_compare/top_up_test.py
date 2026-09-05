#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""补充测试集到 100 条（通用/专业各 50），并去重。"""
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from collections import Counter
from gen_data import generate  # 仅导入函数，不会触发 main()

OUT = os.path.join(os.path.dirname(__file__), "data", "test_100.jsonl")

records = []
with open(OUT, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

train_queries = set()
with open(os.path.join(os.path.dirname(__file__), "data", "train_200.jsonl"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            train_queries.add(json.loads(line)["query"])

cnt = Counter(r["label"] for r in records)
print("补充前:", dict(cnt))

for cat in ("通用知识", "专业咨询"):
    need = 50 - cnt.get(cat, 0)
    if need > 0:
        print(f"补充 {cat} {need} 条 ...")
        new = generate(cat, need, temperature=0.9, test=True)
        seen = {r["query"] for r in records} | train_queries
        added = 0
        for r in new:
            if r["query"] not in seen:
                seen.add(r["query"])
                records.append(r)
                added += 1
            if added >= need:
                break
        cnt = Counter(r["label"] for r in records)
        print(f"  补充后 {cat}={cnt.get(cat,0)}")

with open(OUT, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("最终:", dict(Counter(r["label"] for r in records)), "共", len(records))
