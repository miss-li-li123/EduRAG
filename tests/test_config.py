# -*- coding: utf-8 -*-
"""Config 配置加载逻辑单元测试。"""
import os

from base.config import Config, PROJECT_ROOT


def _write_ini(tmp_path, extra=""):
    """写一份最小可用的 config.ini 到临时目录。"""
    content = (
        "[mysql]\n"
        "host = 10.0.0.1\n"
        "user = tester\n"
        "password = secret\n"
        "database = test_db\n"
        "\n"
        "[redis]\n"
        "host = r.example.com\n"
        "port = 6380\n"
        "password = rpass\n"
        "db = 3\n"
        "\n"
        "[milvus]\n"
        "host = m.example.com\n"
        "port = 19540\n"
        "database_name = milvus_db\n"
        "collection_name = my_collection\n"
        "\n"
        "[llm]\n"
        "model = deepseek-test\n"
        "dashscope_api_key = test-key-123\n"
        "dashscope_base_url = https://api.example.test/v1\n"
        "\n"
        "[retrieval]\n"
        "parent_chunk_size = 1000\n"
        "child_chunk_size = 200\n"
        "chunk_overlap = 30\n"
        "retrieval_k = 7\n"
        "candidate_m = 4\n"
        "\n"
        "[app]\n"
        "valid_sources = [\"ai\", \"java\"]\n"
        "customer_service_phone = 400-000-000\n"
        "\n"
        "[logger]\n"
        "log_file = logs/test.log\n"
    ) + extra
    cfg = tmp_path / "config.ini"
    cfg.write_text(content, encoding="utf-8")
    return str(cfg)


def test_load_custom_config_values(tmp_path):
    conf = Config(config_file=_write_ini(tmp_path))
    assert conf.MYSQL_HOST == "10.0.0.1"
    assert conf.MYSQL_USER == "tester"
    assert conf.MYSQL_PASSWORD == "secret"
    assert conf.REDIS_HOST == "r.example.com"
    assert conf.REDIS_PORT == 6380
    assert conf.REDIS_DB == 3
    assert conf.LLM_MODEL == "deepseek-test"
    assert conf.CHILD_CHUNK_SIZE == 200
    assert conf.CHUNK_OVERLAP == 30
    assert conf.RETRIEVAL_K == 7
    assert conf.CANDIDATE_M == 4


def test_valid_sources_parsed_as_list(tmp_path):
    conf = Config(config_file=_write_ini(tmp_path))
    assert conf.VALID_SOURCES == ["ai", "java"]
    assert isinstance(conf.VALID_SOURCES, list)


def test_missing_sections_use_fallbacks(tmp_path):
    # 空配置文件（无任何 section）也应能正常构建，且使用默认值
    empty = tmp_path / "empty.ini"
    empty.write_text("", encoding="utf-8")
    conf = Config(config_file=str(empty))
    assert conf.MYSQL_HOST == "localhost"
    assert conf.REDIS_PORT == 6379
    assert conf.LLM_MODEL == "deepseek-v4-flash"
    assert conf.RETRIEVAL_K == 5
    assert conf.VALID_SOURCES == ["ai", "java", "test", "ops", "bigdata"]
    assert conf.CUSTOMER_SERVICE_PHONE == "12345678"


def test_log_file_resolved_to_absolute(tmp_path):
    conf = Config(config_file=_write_ini(tmp_path))
    assert os.path.isabs(conf.LOG_FILE)
    # Windows 下 join 产生 \\ ，endswith 需容忍分隔符差异
    expected = os.path.normpath(os.path.join("logs", "test.log"))
    assert os.path.normpath(conf.LOG_FILE).endswith(expected)
    # 相对路径会锚定到项目根目录
    assert conf.LOG_FILE.startswith(PROJECT_ROOT)


def test_dashscope_api_key_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ALIYUN_API_KEY", "ENV-KEY-999")
    conf = Config(config_file=_write_ini(tmp_path))
    assert conf.DASHSCOPE_API_KEY == "ENV-KEY-999"


def test_default_abs_paths(tmp_path):
    conf = Config(config_file=_write_ini(tmp_path))
    # model 路径锚定在项目目录内
    assert os.path.isabs(conf.nlp_bert_doc_seg)
    assert "rag_qa" in conf.nlp_bert_doc_seg
