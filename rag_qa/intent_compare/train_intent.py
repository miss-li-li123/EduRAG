#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""以 bert_query_classifier 为基座，用 train_200.jsonl 微调意图分类 BERT（原生 torch 循环，免 accelerate）。

输出：rag_qa/core/bert_results/intent-ckpt/
"""
import sys, os, json, argparse
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

LABEL_MAP = {"通用知识": 0, "专业咨询": 1}
ID2LABEL = {0: "通用知识", 1: "专业咨询"}
MAX_LEN = 128


def load_data(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            texts.append(d["query"])
            labels.append(LABEL_MAP[d["label"]])
    return texts, labels


def build_dataset(texts, labels, tokenizer):
    enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    return TensorDataset(
        enc["input_ids"], enc["attention_mask"], enc["token_type_ids"], torch.tensor(labels)
    )


def evaluate(model, dl, device):
    model.eval()
    correct = total = 0
    val_loss = 0.0
    with torch.no_grad():
        for ids, mask, tt, y in dl:
            ids, mask, tt, y = ids.to(device), mask.to(device), tt.to(device), y.to(device)
            out = model(input_ids=ids, attention_mask=mask, token_type_ids=tt, labels=y)
            val_loss += out.loss.item()
            preds = out.logits.argmax(-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total, val_loss / max(1, len(dl))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "data", "train_200.jsonl"))
    ap.add_argument("--base_model", default=os.path.join(ROOT, "rag_qa", "core", "bert_query_classifier"))
    ap.add_argument("--output", default=os.path.join(ROOT, "rag_qa", "core", "bert_results", "intent-ckpt"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, base={args.base_model}")

    texts, labels = load_data(args.data)
    print(f"数据量={len(texts)}  标签分布={ {ID2LABEL[k]: v for k, v in Counter(labels).items()} }")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tokenizer = BertTokenizer.from_pretrained(args.base_model)
    model = BertForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()}
    )
    model.to(device)

    train_ds = build_dataset(train_texts, train_labels, tokenizer)
    val_ds = build_dataset(val_texts, val_labels, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    best_acc = -1.0
    os.makedirs(args.output, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for ids, mask, tt, y in train_dl:
            ids, mask, tt, y = ids.to(device), mask.to(device), tt.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, token_type_ids=tt, labels=y)
            loss = out.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        acc, vloss = evaluate(model, val_dl, device)
        print(f"epoch {epoch}/{args.epochs}  train_loss={total_loss/len(train_dl):.4f}  "
              f"val_loss={vloss:.4f}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print(f"  -> 保存最佳模型 (val_acc={acc:.4f})")

    if best_acc < 0:
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)

    print(f"训练完成。最佳验证准确率={best_acc:.4f}，模型目录={args.output}")


if __name__ == "__main__":
    main()
