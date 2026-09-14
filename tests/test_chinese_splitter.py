# -*- coding: utf-8 -*-
"""中文递归文本切分器单元测试。"""
from rag_qa.edu_text_spliter import ChineseRecursiveTextSplitter


def test_split_short_text_kept_single():
    splitter = ChineseRecursiveTextSplitter(chunk_size=200, chunk_overlap=0)
    text = "黑马程序员AI课程深受欢迎。"
    chunks = splitter.split_text(text)
    assert chunks == [text]


def test_split_long_text_produces_multiple_chunks():
    # 超过 chunk_size 的长文本应被切成多块
    splitter = ChineseRecursiveTextSplitter(chunk_size=50, chunk_overlap=5)
    text = "第一句话讲了基础概念。" * 40
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    assert all(c.strip() != "" for c in chunks)


def test_split_by_sentence_boundary():
    # 中文逗号/句号应作为切分边界
    splitter = ChineseRecursiveTextSplitter(chunk_size=100, chunk_overlap=0)
    text = "人工智能在教育领域应用广泛。机器学习是重要分支。"
    chunks = splitter.split_text(text)
    # 每个短句应被保留在某个 chunk 里
    joined = "".join(chunks)
    assert "人工智能在教育领域应用广泛" in joined
    assert "机器学习是重要分支" in joined


def test_no_trailing_blank_chunks():
    splitter = ChineseRecursiveTextSplitter(chunk_size=30, chunk_overlap=3)
    text = "a\n\nb\n\nc\n\n"
    chunks = splitter.split_text(text)
    assert all(c.strip() != "" for c in chunks)
    assert all(isinstance(c, str) for c in chunks)


def test_empty_text_returns_empty():
    splitter = ChineseRecursiveTextSplitter(chunk_size=50, chunk_overlap=0)
    assert splitter.split_text("") == []
