# -*- coding: utf-8 -*-
"""rag_qa/core/prompts 提示词模板单元测试。"""
import pytest

from rag_qa.core.prompts import RAGPrompts


def test_rag_prompt_format():
    prompt = RAGPrompts.rag_prompt()
    out = prompt.format(
        context="黑马程序员课程",
        history="user: 你好\nai: 你好",
        question="AI学费多少",
        phone="400-123-456",
    )
    assert "黑马程序员课程" in out
    assert "你好" in out
    assert "AI学费多少" in out
    assert "400-123-456" in out


def test_rag_prompt_missing_phone_raises():
    prompt = RAGPrompts.rag_prompt()
    with pytest.raises(KeyError):
        # 缺少 phone 时 format 应抛 KeyError，保证模板变量完整
        prompt.format(context="ctx", history="his", question="q?")


def test_hyde_prompt():
    prompt = RAGPrompts.hyde_prompt()
    out = prompt.format(query="AI在教育中的应用")
    assert "AI在教育中的应用" in out


def test_subquery_prompt():
    prompt = RAGPrompts.subquery_prompt()
    out = prompt.format(query="对比Milvus和Redis的优缺点")
    assert "对比Milvus和Redis的优缺点" in out


def test_backtracking_prompt():
    prompt = RAGPrompts.backtracking_prompt()
    out = prompt.format(query="100亿条数据如何存储")
    assert "100亿条数据如何存储" in out


def test_all_prompts_have_required_vars():
    # 每个模板声明的输入变量应能成功 format
    cases = [
        ("rag", ["context", "history", "question", "phone"],
         dict(context="c", history="h", question="q", phone="p")),
        ("hyde", ["query"], dict(query="q")),
        ("subquery", ["query"], dict(query="q")),
        ("backtracking", ["query"], dict(query="q")),
    ]
    for name, _, kwargs in cases:
        fn = getattr(RAGPrompts, f"{name}_prompt")
        prompt = fn()
        out = prompt.format(**kwargs)
        assert isinstance(out, str) and out
