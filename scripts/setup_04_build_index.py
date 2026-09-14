# -*- coding: utf-8 -*-
"""遍历 VALID_SOURCES 建 Milvus 索引"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from base import Config
from rag_qa.core.document_loader import process_documents
from rag_qa.core.vector_store import VectorStore
from pymilvus import MilvusClient

conf = Config()

# 确保 Milvus database 存在
_admin = MilvusClient(uri=f"http://{conf.MILVUS_HOST}:{conf.MILVUS_PORT}")
try:
    dbs = list(_admin.list_databases())
    if conf.MILVUS_DATABASE_NAME not in dbs:
        _admin.create_database(conf.MILVUS_DATABASE_NAME)
        print(f"已创建 Milvus 数据库: {conf.MILVUS_DATABASE_NAME}")
    else:
        print(f"Milvus 数据库已存在: {conf.MILVUS_DATABASE_NAME}")
finally:
    _admin.close()

vs = VectorStore()
data_root = os.path.join(ROOT, "rag_qa", "data")
total = 0
for source in conf.VALID_SOURCES:
    d = os.path.join(data_root, f"{source}_data")
    if not os.path.isdir(d):
        print(f"跳过（不存在）: {d}")
        continue
    chunks = process_documents(d, conf.PARENT_CHUNK_SIZE, conf.CHILD_CHUNK_SIZE, conf.CHUNK_OVERLAP)
    if chunks:
        vs.add_documents(chunks)
        total += len(chunks)
        print(f"{source}: 写入 {len(chunks)} 个子块")
print(f"完成，共 {total} 个子块")
