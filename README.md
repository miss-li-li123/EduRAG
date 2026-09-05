# IT教育智能问答系统（EduRAG）

面向 IT 教育培训场景的 AI 问答系统，融合 **MySQL/BM25 精确问答** 与 **Milvus 向量 RAG** 两条链路，由 **BERT 意图分类器** 自动分流（通用知识 / 专业咨询），前端经 FastAPI（REST + SSE）提供流式问答。

## 核心能力

- 双路召回：高频标准问题 BM25 精确命中，长尾问题走 RAG 检索增强
- 多策略检索增强：直接检索 / HyDE / 子查询 / 回溯问题检索（LLM 动态选择）
- BGE-M3 稠密+稀疏混合检索 + BGE-Reranker 重排
- 对话历史记忆（MySQL，最近 5 轮）、学科过滤、多格式文档入库（含 OCR）
- 意图识别对比实验：BERT 微调（0.97）vs 大模型+提示词（0.89）

## 文档

详细文档位于 [`docs/`](./docs/) 目录：

- [项目概述](./docs/01-项目概述.md)
- [系统架构](./docs/02-系统架构.md)
- [模块说明](./docs/03-模块说明.md)
- [快速开始与部署](./docs/04-快速开始与部署.md)
- [API 接口文档](./docs/05-API接口文档.md)
- [意图识别对比实验](./docs/06-意图识别对比实验.md)
- [目录结构与文件清单](./docs/07-目录结构与文件清单.md)

## 快速开始

```bash
pip install -r requirements-windows.txt
export ALIYUN_API_KEY="你的DashScope_API_Key"
cd docker/milvus_redis && docker-compose up -d
uvicorn app:app --host 0.0.0.0 --port 8080
```
