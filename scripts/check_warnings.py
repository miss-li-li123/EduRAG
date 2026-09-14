# -*- coding: utf-8 -*-
"""以警告即异常的方式冒烟测试核心链路"""
import warnings
# 第三方库（jieba/pkg_resources、torch/transformers 等）的弃用告警不在项目可控范围，
# 只对项目自身代码触发的告警严格处理
warnings.filterwarnings("error", message=".*", category=Warning,
                        module=r"(app|main|base|mysql_qa|rag_qa|scripts)(\..*)?")
warnings.filterwarnings("default", category=DeprecationWarning)
warnings.filterwarnings("default", category=FutureWarning)
warnings.filterwarnings("default", category=UserWarning)

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

print(">> 导入 app（触发全部模块初始化）...")
import app  # noqa
print(">> app 导入成功，无警告")

print(">> BM25 链路 ...")
ans, need_rag = app.qa_system.bm25_search.search("windows如何安装redis")
assert ans and not need_rag
print("   OK")

print(">> 意图分类 ...")
cat = app.qa_system.rag_system.query_classifier.predict_category("Python 培训班学费多少")
print("   ->", cat)

print(">> 向量检索（含 reranker）...")
docs = app.qa_system.vector_store.hybrid_search_with_rerank("人工智能课程岗位", k=2)
print("   检索到", len(docs), "个父文档")
print("ALL CLEAN")
