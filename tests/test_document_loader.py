# -*- coding: utf-8 -*-
"""rag_qa/core/document_loader 文档加载与分层切分集成测试。

不依赖真实文件系统内容，仅验证：
- 加载器注册表覆盖常见文件类型
- 目录名到 source 学科的映射规则
- 分层切分生成正确的 id / parent_id / parent_content 元数据
- 按扩展名选择 Markdown 或中文切分器
"""
import os


import rag_qa.core.document_loader as dl

SUPPORTED = {".txt", ".pdf", ".docx", ".ppt", ".pptx", ".jpg", ".png", ".md"}


def test_document_loader_registry_covers_common_types():
    assert SUPPORTED.issubset(set(dl.document_loaders.keys()))
    assert ".txt" in dl.document_loaders


def test_source_name_from_directory():
    # ai_data -> ai
    docs = dl.load_documents_from_directory(
        os.path.join(os.path.dirname(__file__), "fixtures", "ai_data")
    )
    assert all(d.metadata["source"] == "ai" for d in docs)
    assert all("file_path" in d.metadata for d in docs)
    assert all("timestamp" in d.metadata for d in docs)


def test_process_documents_sets_parent_child_ids(tmp_path):
    # 构造一个临时 .txt 文档目录
    src = tmp_path / "java_data"
    src.mkdir()
    (src / "intro.txt").write_text(
        "Java是一门面向对象的编程语言。它被广泛应用于企业级开发。"
        "集合框架提供了List、Map等常用数据结构。",
        encoding="utf-8",
    )

    chunks = dl.process_documents(
        str(src), parent_chunk_size=20, child_chunk_size=50, chunk_overlap=0
    )
    assert chunks, "应当产生至少一个子块"

    first = chunks[0]
    assert first.metadata["source"] == "java"
    # id 形如 doc_0_parent_0_child_0
    assert first.metadata["id"].startswith("doc_0_parent_")
    assert "_child_" in first.metadata["id"]
    assert first.metadata["parent_id"] in first.metadata["id"]
    # 子块应携带父块全文
    assert first.metadata["parent_content"]


def test_process_documents_namespaced_ids(tmp_path):
    src = tmp_path / "ai_data"
    src.mkdir()
    (src / "a1.txt").write_text("第一章：索引基础。" * 30, encoding="utf-8")
    (src / "a2.txt").write_text("第二章：向量检索。" * 30, encoding="utf-8")

    docs = dl.load_documents_from_directory(str(src))
    assert len(docs) == 2

    chunks = dl.process_documents(str(src), 60, 40, 0)
    ids = [c.metadata["id"] for c in chunks]
    # 不同文档的父块 id 前缀不同（doc_0_ / doc_1_），保证唯一命名空间
    prefixes = {i.split("parent")[0] for i in ids}
    assert len(prefixes) == 2
