#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 大模型+提示词 对测试集做意图预测，逐条即时落盘到 data/llm_pred.jsonl（与测试集同序）。"""
import sys, os, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from llm_intent_classifier import LLMIntentClassifier

HERE = os.path.dirname(__file__)
TEST = os.path.join(HERE, "data", "test_100.jsonl")
OUT = os.path.join(HERE, "data", "llm_pred.jsonl")


def main():
    rows = []
    with open(TEST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    clf = LLMIntentClassifier()
    correct = 0
    with open(OUT, "w", encoding="utf-8") as fout:
        for i, r in enumerate(rows, 1):
            pred = clf.predict(r["query"])
            ok = (pred == r["label"])
            correct += ok
            fout.write(json.dumps({"query": r["query"], "gold": r["label"], "pred": pred},
                                  ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{i}/{len(rows)}] gold={r['label']} pred={pred} {'OK' if ok else 'X'}", flush=True)
    print(f"\n完成，缓存: {OUT}  即时准确率(基于gold)={correct/len(rows):.4f}", flush=True)


if __name__ == "__main__":
    main()
