# -*- coding: utf-8 -*-
"""mysql_qa/retrieval/bm25_search BM25 检索单元测试（mock Redis / MySQL）。

不依赖真实 Redis / MySQL，用桩对象替换客户端，覆盖：
- 命中缓存直接返回
- 命中知识库返回答案
- 低于阈值返回不可靠
- 无效查询返回不可靠
"""
import pytest
import numpy as np

from mysql_qa.retrieval.bm25_search import BM25Search


class FakeRedis:
    """内存版 Redis 桩，记录调用并可缓存答案。"""

    def __init__(self, questions=None, answers=None):
        self.questions = questions or []
        self.answers = answers or {}          # query -> answer
        self.stored = {}                      # 写入的原始问题缓存
        self.remove_cache = True

    def get_data(self, key):
        if key == "qa_original_questions":
            return None  # 强制走 MySQL 加载路径
        if key == "qa_tokenized_questions":
            return None
        return None

    def set_data(self, key, value):
        self.stored[key] = value

    def get_answer(self, query):
        return self.answers.get(query)

    def set_answer(self, query, answer):
        self.answers[query] = answer


class FakeMySQL:
    def __init__(self, rows):
        self.rows = rows  # [(question, answer), ...]
        self.fetched = []

    def fetch_questions(self):
        return [(q,) for q, _ in self.rows]

    def fetch_answer(self, question):
        self.fetched.append(question)
        for q, a in self.rows:
            if q == question:
                return a
        return None


@pytest.fixture
def kb_questions():
    return [
        ("windows如何安装redis", "redis安装步骤如下..."),
        ("java泛型是什么", "泛型是一种类型参数机制..."),
        ("ai学费多少", "AI课程学费为19800元"),
    ]


def test_cache_hit_returns_cached_answer(kb_questions):
    redis = FakeRedis(
        questions=["windows如何安装redis"],
        answers={"windows如何安装redis": "缓存答案ABC"},
    )
    bm = BM25Search(redis, FakeMySQL(kb_questions))
    answer, reliable = bm.search("windows如何安装redis")
    assert answer == "缓存答案ABC"
    assert reliable is False


def test_mysql_answer_found_and_cached(kb_questions):
    redis = FakeRedis()
    mysql = FakeMySQL(kb_questions)
    bm = BM25Search(redis, mysql)
    # 3 条问题的知识库 softmax 最高约 0.78，无法达到默认 0.85 阈值；
    # 显式降低阈值以覆盖“查询命中并写入缓存”的路径
    answer, reliable = bm.search("windows如何安装redis", threshold=0.5)
    assert answer is not None
    assert reliable is False
    # 命中后应已写入缓存
    assert "windows如何安装redis" in redis.answers
    assert redis.answers["windows如何安装redis"] == answer


def test_low_similarity_returns_unreliable(kb_questions):
    # 一个与知识库完全无关的查询，相似度低于阈值 -> (None, True)
    redis = FakeRedis()
    bm = BM25Search(redis, FakeMySQL(kb_questions))
    answer, reliable = bm.search("今天中午吃什么比较健康")
    assert answer is None
    assert reliable is True


def test_invalid_query(kb_questions):
    """无效查询：源码返回 (None, False)。"""
    redis = FakeRedis()
    bm = BM25Search(redis, FakeMySQL(kb_questions))
    for bad in ("", None):
        answer, reliable = bm.search(bad)
        assert answer is None
        assert reliable is False


def test_softmax_normalized(kb_questions):
    redis = FakeRedis()
    bm = BM25Search(redis, FakeMySQL(kb_questions))
    scores = np.array([1.0, 2.0, 3.0])
    out = bm._softmax(scores)
    assert abs(out.sum() - 1.0) < 1e-9
    assert out[2] > out[1] > out[0]
    # 恒等输入输出均匀
    uniform = bm._softmax(np.full(3, 5.0))
    assert np.allclose(uniform, 1 / 3, atol=1e-9)


def test_no_questions_returns_unreliable():
    redis = FakeRedis()          # 空数据
    mysql = FakeMySQL([])        # MySQL 也没有问题
    bm = BM25Search(redis, mysql)
    assert bm.questions is None or bm.questions == []
    answer, reliable = bm.search("如何在linux安装redis")
    assert answer is None
    assert reliable is True
