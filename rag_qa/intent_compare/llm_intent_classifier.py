#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""大模型 + 提示词 的意图分类器（通用知识 / 专业咨询）。"""
import sys, os, re, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from base import Config
from openai import OpenAI

conf = Config()
client = OpenAI(api_key=conf.DASHSCOPE_API_KEY, base_url=conf.DASHSCOPE_BASE_URL, timeout=60)
MODEL = conf.LLM_MODEL

SYSTEM_PROMPT = (
    "你是一个高效的文本分类系统，任务是判断用户的查询属于“通用知识”还是“专业咨询”。\n"
    "## 类别定义\n"
    "- “通用知识”：数学计算、代码生成/纠错、编程概念与原理、计算机基础常识，以及其他自然科学/生活常识问题。\n"
    "- “专业咨询”：与IT教育培训相关的查询，如课程详情、师资介绍、项目内容、培训周期、地点、时间、费用/学费、分期、试听、就业保障、开班日期、校区城市、证书等。\n"
    "## 输出要求\n"
    "1. 仅输出一个 JSON 对象：{\"query\": \"原始查询\", \"label\": \"通用知识\"} 或 {\"label\": \"专业咨询\"}。\n"
    "2. 标签只能是“通用知识”或“专业咨询”。\n"
    "3. 不要输出除 JSON 之外的任何文字。\n"
    "4. 若存在歧义，优先判断是否涉及具体的IT教育培训服务，若是则分类为“专业咨询”。"
)


class LLMIntentClassifier:
    def __init__(self, model=MODEL, temperature=0.0, max_retry=3):
        self.model = model
        self.temperature = temperature
        self.max_retry = max_retry

    def predict(self, query):
        for _ in range(self.max_retry):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"请分类：{query}"},
                    ],
                    temperature=self.temperature,
                    max_tokens=128,
                )
                text = (resp.choices[0].message.content or "").strip()
                label = self._parse(text, query)
                if label:
                    return label
            except Exception as e:
                print(f"  [warn] LLM 调用失败: {e}")
        # 兜底
        return "通用知识"

    @staticmethod
    def _parse(text, query):
        # 1) 直接 JSON
        try:
            d = json.loads(text)
            if isinstance(d, dict) and d.get("label") in ("通用知识", "专业咨询"):
                return d["label"]
        except Exception:
            pass
        # 2) 正则提取标签
        m = re.search(r"(通用知识|专业咨询)", text)
        if m:
            return m.group(1)
        return None


if __name__ == "__main__":
    clf = LLMIntentClassifier()
    for q in ["5*9等于多少？", "AI培训有哪些老师？", "Python怎么读文件？"]:
        print(q, "->", clf.predict(q))
