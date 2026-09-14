#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""调用大模型生成意图识别训练/测试数据（通用知识 vs 专业咨询）。"""
import sys
import os
import json
import re
import time
import argparse

# 将项目根目录加入 sys.path，以便 import base / rag_qa
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from base import Config
from openai import OpenAI

conf = Config()
client = OpenAI(api_key=conf.DASHSCOPE_API_KEY, base_url=conf.DASHSCOPE_BASE_URL)
MODEL = conf.LLM_MODEL  # deepseek-v4-flash


def parse_records(text):
    """从模型输出中鲁棒地解析出 record 列表。"""
    text = text.strip()
    # 去掉可能的 ```json 代码块
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = text.rstrip("`").strip()
    # 尝试整体 JSON 数组
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict) and d.get("query")]
    except Exception:
        pass
    # 截取 [ ... ]
    s, e = text.find("["), text.rfind("]")
    if s != -1 and e != -1:
        try:
            data = json.loads(text[s:e + 1])
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict) and d.get("query")]
        except Exception:
            pass
    # 逐行解析
    recs = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            d = json.loads(line)
            if isinstance(d, dict) and d.get("query"):
                recs.append(d)
        except Exception:
            pass
    return recs


def generate(category, n, batch=50, temperature=0.95, test=False):
    """生成 n 条指定类别的数据，分批次请求直到凑够。"""
    if category == "专业咨询":
        domain_hint = (
            "覆盖多种学科方向，至少包含：人工智能(AI)、Java、软件测试、运维/云计算、"
            "大数据、前端开发、嵌入式、UI设计、网络安全、鸿蒙/移动开发 等。"
        )
        rule = (
            "『专业咨询』类专指与IT教育培训相关的查询，例如：课程详情、师资介绍、"
            "项目内容、培训周期、上课地点、时间、费用/学费、分期、试听、就业保障、"
            "开班日期、校区城市、证书等。" + domain_hint +
            "不要包含数学计算/纯代码编写/通用概念解释（那属于通用知识）。"
        )
    else:
        rule = (
            "『通用知识』类主要包含：数学计算、代码生成/纠错、编程概念与原理、"
            "计算机基础常识、以及其他自然科学/生活常识问题。"
            "注意：不要涉及任何具体的IT教育培训机构、课程、学费、师资等内容"
            "（那属于专业咨询）。"
        )

    extra = ""
    if test:
        extra = ("另外，请尽量生成一些边界/易混淆的句子（例如既像通用知识又像专业咨询的"
                 "模糊问法），使数据更具挑战性；并避免与常见模板重复。")

    collected = []
    seen = set()
    attempts = 0
    while len(collected) < n and attempts < 20:
        need = min(batch, n - len(collected))
        prompt = (
            f"你是一个数据标注专家。请生成关于「{category}」类用户提问的查询句子。\n"
            f"定义：{rule}\n"
            f"要求：\n"
            f"1. 每条是一个 JSON 对象，格式 {{\"query\": \"用户问句\", \"label\": \"{category}\"}}。\n"
            f"2. 句子要自然、口语化、多样化，避免重复句式；可包含口语化、错别字、省略等真实用户输入风格。\n"
            f"3. 直接输出 JSON 数组，不要输出任何解释文字，不要使用代码块标记。\n"
            f"{extra}\n"
            f"请生成 {need} 条。"
        )
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=2500,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  [warn] API 调用失败: {e}")
            time.sleep(3)
            attempts += 1
            continue

        recs = parse_records(text)
        for r in recs:
            q = str(r.get("query", "")).strip()
            lab = category  # 强制类别，忽略模型可能写错的 label
            if q and q not in seen:
                seen.add(q)
                collected.append({"query": q, "label": lab})
        attempts += 1
        print(f"  {category}: 已收集 {len(collected)}/{n} (本批解析 {len(recs)})")
        time.sleep(0.5)

    if len(collected) < n:
        print(f"  [warn] {category} 仅生成 {len(collected)}/{n} 条")
    return collected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_n", type=int, default=200, help="训练集总条数(默认200, 50/50)")
    ap.add_argument("--test_n", type=int, default=100, help="测试集总条数(默认100, 50/50)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[1/2] 生成训练集 {args.train_n} 条 (50/50) ...")
    train_gen = generate("通用知识", args.train_n // 2, temperature=0.95)
    train_pro = generate("专业咨询", args.train_n // 2, temperature=0.95)
    train = train_gen + train_pro

    print(f"[2/2] 生成测试集 {args.test_n} 条 (50/50) ...")
    test_gen = generate("通用知识", args.test_n // 2, temperature=0.9, test=True)
    test_pro = generate("专业咨询", args.test_n // 2, temperature=0.9, test=True)
    test = test_gen + test_pro

    # 去重：测试集与训练集不能有完全相同的 query
    train_q = {t["query"] for t in train}
    before = len(test)
    test = [t for t in test if t["query"] not in train_q]
    removed = before - len(test)
    if removed:
        print(f"  测试集去重移除 {removed} 条与训练集重复的样本，补充生成...")
        need_gen = (args.test_n // 2) - sum(1 for t in test if t["label"] == "通用知识")
        need_pro = (args.test_n // 2) - sum(1 for t in test if t["label"] == "专业咨询")
        if need_gen > 0:
            test += generate("通用知识", need_gen, temperature=0.9, test=True)
        if need_pro > 0:
            test += generate("专业咨询", need_pro, temperature=0.9, test=True)
        # 再次去重
        seen2 = set(train_q)
        dedup = []
        for t in test:
            if t["query"] not in seen2:
                seen2.add(t["query"])
                dedup.append(t)
        test = dedup

    train_path = os.path.join(args.out, "train_200.jsonl")
    test_path = os.path.join(args.out, "test_100.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(test_path, "w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计
    def stat(records):
        from collections import Counter
        c = Counter(r["label"] for r in records)
        return dict(c)

    print("\n=== 生成完成 ===")
    print(f"训练集: {train_path}  共 {len(train)} 条 {stat(train)}")
    print(f"测试集: {test_path}  共 {len(test)} 条 {stat(test)}")


if __name__ == "__main__":
    main()
