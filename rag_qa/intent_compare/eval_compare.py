#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在同一测试集上对比 BERT 微调 与 大模型+提示词 的意图识别准确率。"""
import sys, os, json, argparse
from collections import Counter
import numpy as np
import torch; torch.set_num_threads(1)  # 避免 Windows 上 OpenMP 多线程导致的 segfault
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))  # 确保同目录的 llm_intent_classifier 可导入


def load_query_classifier(model_path):
    """文件级加载 QueryClassifier，绕过 rag_qa 包 __init__（避免 milvus 依赖）
    与 transformers.Trainer 触发 accelerate 在 Windows 上的 segfault 问题。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "query_classifier_loaded",
        os.path.join(ROOT, "rag_qa", "core", "query_classifier.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.QueryClassifier(model_path=model_path)


from llm_intent_classifier import LLMIntentClassifier

CLASSES = ["通用知识", "专业咨询"]


def load_test(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(preds, golds):
    acc = accuracy_score(golds, preds)
    cm = confusion_matrix(golds, preds, labels=CLASSES)
    rep = classification_report(golds, preds, labels=CLASSES, output_dict=True, zero_division=0)
    macro_f1 = f1_score(golds, preds, labels=CLASSES, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": macro_f1, "confusion_matrix": cm.tolist(),
            "per_class": {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"],
                              "f1": rep[c]["f1-score"], "support": rep[c]["support"]} for c in CLASSES},
            "report_text": classification_report(golds, preds, labels=CLASSES, zero_division=0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=os.path.join(os.path.dirname(__file__), "data", "test_100.jsonl"))
    ap.add_argument("--bert_model", default=os.path.join(ROOT, "rag_qa", "core", "bert_results", "intent-ckpt"))
    args = ap.parse_args()
    out_dir = os.path.dirname(args.test)

    rows = load_test(args.test)
    golds = [r["label"] for r in rows]
    queries = [r["query"] for r in rows]
    print(f"测试集: {len(rows)} 条  分布={dict(Counter(golds))}")

    # ---- BERT ----
    print("\n[1] BERT 微调模型预测中 ...")
    bert = load_query_classifier(args.bert_model)
    bert_preds = [bert.predict_category(q) for q in queries]
    bert_res = evaluate(bert_preds, golds)
    print("BERT 准确率: {:.4f}  Macro-F1: {:.4f}".format(bert_res["accuracy"], bert_res["macro_f1"]))

    # ---- LLM + prompt ----
    print("\n[2] 大模型+提示词 预测中 ...")
    llm_model_name = LLMIntentClassifier().model  # 仅取模型名，不发起 API 请求
    llm_cache = os.path.join(out_dir, "llm_pred.jsonl")
    if os.path.exists(llm_cache):
        print(f"  复用缓存: {llm_cache}")
        llm_preds = [json.loads(l)["pred"] for l in open(llm_cache, encoding="utf-8")]
    else:
        llm = LLMIntentClassifier()
        llm_preds = [llm.predict(q) for q in queries]
    llm_res = evaluate(llm_preds, golds)
    print("LLM+提示词 准确率: {:.4f}  Macro-F1: {:.4f}".format(llm_res["accuracy"], llm_res["macro_f1"]))

    # ---- 对比报告 ----
    out_dir = os.path.dirname(args.test)
    report = {
        "test_size": len(rows),
        "test_distribution": dict(Counter(golds)),
        "bert": {
            "model_path": args.bert_model,
            "accuracy": bert_res["accuracy"],
            "macro_f1": bert_res["macro_f1"],
            "confusion_matrix": bert_res["confusion_matrix"],
            "per_class": bert_res["per_class"],
        },
        "llm_prompt": {
            "model": llm_model_name,
            "accuracy": llm_res["accuracy"],
            "macro_f1": llm_res["macro_f1"],
            "confusion_matrix": llm_res["confusion_matrix"],
            "per_class": llm_res["per_class"],
        },
        "diff_accuracy": llm_res["accuracy"] - bert_res["accuracy"],
    }
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md = f"""# 意图识别对比报告：BERT 微调 vs 大模型+提示词

- 测试集：{len(rows)} 条（{dict(Counter(golds))}）
- 任务：二分类（通用知识 / 专业咨询）

## 核心指标

| 方法 | 准确率 | Macro-F1 |
|------|--------|----------|
| BERT 微调（`{os.path.basename(args.bert_model)}`） | {bert_res['accuracy']:.4f} | {bert_res['macro_f1']:.4f} |
| 大模型+提示词（{llm_model_name}） | {llm_res['accuracy']:.4f} | {llm_res['macro_f1']:.4f} |

准确率差值（LLM - BERT）：{report['diff_accuracy']:+.4f}

## BERT 混淆矩阵（行=真实，列=预测）

```
{' / '.join(CLASSES)}
"""
    for i, c in enumerate(CLASSES):
        md += f"{c}: {bert_res['confusion_matrix'][i]}\n"
    md += f"""
## 大模型+提示词 混淆矩阵（行=真实，列=预测）

```
{' / '.join(CLASSES)}
"""
    for i, c in enumerate(CLASSES):
        md += f"{c}: {llm_res['confusion_matrix'][i]}\n"
    md += f"""
## BERT 逐类指标
{bert_res['report_text']}
## 大模型+提示词 逐类指标
{llm_res['report_text']}
"""

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)

    print("\n=== 对比结果 ===")
    print(f"BERT 准确率: {bert_res['accuracy']:.4f} | LLM+提示词 准确率: {llm_res['accuracy']:.4f}")
    print(f"报告已保存: {os.path.join(out_dir, 'report.json')} / report.md")


if __name__ == "__main__":
    main()
