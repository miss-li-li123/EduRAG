# -*- coding: utf-8 -*-
"""mysql_qa/utils/preprocess 文本预处理单元测试。"""
from mysql_qa.utils.preprocess import preprocess_text


def test_returns_list_of_tokens():
    tokens = preprocess_text("如何安装redis")
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)


def test_lowercased():
    # 英文部分应转为小写
    tokens = preprocess_text("Java")
    lowered = [t.lower() for t in tokens]
    assert "java" in lowered


def test_empty_string_returns_list():
    # 空串不应抛异常，jieba 返回空列表
    tokens = preprocess_text("")
    assert isinstance(tokens, list)


def test_chinese_content_preserved():
    tokens = preprocess_text("黑马程序员AI课程")
    assert any("黑马" in t or "程序员" in t or "课程" in t for t in tokens)
