from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from pydantic import BaseModel
import asyncio
import json
import uuid
from typing import Optional
import time
import re

# 导入现有的系统
from main import IntegratedQASystem

# 创建应用实例
app = FastAPI(title="问答系统API", description="集成MySQL和RAG的智能问答系统")

# 配置CORS，允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建静态文件目录
os.makedirs("static", exist_ok=True)

# 创建全局QA系统实例
qa_system = IntegratedQASystem()

# 定义日常问候用语模式和回复
GREETING_PATTERNS = [
    {
        "pattern": r"^(你好|您好|hi|hello)",
        "response": "你好！我是黑马程序员，专注于为学生答疑解惑，很高兴为你服务！"
    },
    {
        "pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",
        "response": "我是黑马程序员，你的智能学习助手，致力于提供 IT 教育相关的解答！"
    },
    {
        "pattern": r"^(在吗|在不在|有人吗)",
        "response": "我在！我是黑马程序员，随时为你解答问题！"
    },
    {
        "pattern": r"^(干嘛呢|你在干嘛|做什么)",
        "response": "我正在待命，随时为你解答 IT 学习相关的问题！有什么我可以帮你的？"
    }
]


# 定义请求模型
class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None


# 定义响应模型
class QueryResponse(BaseModel):
    answer: str
    is_streaming: bool
    session_id: str
    processing_time: float


# 添加静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")


# 根路径重定向到index.html
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


# 创建新会话
@app.post("/api/create_session")
async def create_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


# 查询历史消息
@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    try:
        history = qa_system.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")


# 清除历史消息
@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    success = qa_system.clear_session_history(session_id)
    if success:
        return {"status": "success", "message": "历史记录已清除"}
    else:
        raise HTTPException(status_code=500, detail="清除历史记录失败")

# 检查是否为日常问候并返回模板回复
def check_greeting(query: str) -> Optional[str]:
    query_text = query.strip()
    for pattern_info in GREETING_PATTERNS:
        if re.match(pattern_info["pattern"], query_text, re.IGNORECASE):
            return pattern_info["response"]
    return None


def _sse(event: dict) -> str:
    """将事件字典包装为 SSE data 帧"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# 入参 出参
# 非流式查询接口
@app.post("/api/query")
async def query(request: QueryRequest):
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    # 检查是否为日常问候
    greeting_response = check_greeting(request.query)
    if greeting_response:
        return {
            "answer": greeting_response,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }

    # 判断是否需要流式处理（基于 need_rag）
    answer, need_rag = qa_system.bm25_search.search(request.query, threshold=0.85)
    if need_rag:
        # 需要 RAG 和 LLM 处理，返回流式响应提示
        return {
            "answer": "请使用WebSocket接口获取流式响应",
            "is_streaming": True,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }

    # 非流式查询，直接返回 BM25 检索的答案
    return {
        "answer": answer,
        "is_streaming": False,
        "session_id": session_id,
        "processing_time": time.time() - start_time
    }


# SSE 流式查询生成器
async def generate_sse_stream(query: str, source_filter: Optional[str], session_id: str):
    """生成 SSE 格式的流式响应"""
    start_time = time.time()  # 记录开始时间

    try:
        # 发送开始事件
        yield _sse({'type': 'start', 'session_id': session_id})

        # 检查是否为日常问候
        greeting_response = check_greeting(query)
        if greeting_response:
            # 发送问候回复
            yield _sse({'type': 'token', 'token': greeting_response, 'session_id': session_id})
            # 发送结束事件
            yield _sse({'type': 'end', 'session_id': session_id, 'is_complete': True,
                        'processing_time': time.time() - start_time})
            return

        # 调用问答系统，流式处理查询
        collected_answer = ""
        for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
            collected_answer += token  # 累积答案

            if is_complete and not collected_answer:
                # 发送结束事件
                yield _sse({'type': 'end', 'session_id': session_id, 'is_complete': True,
                           'processing_time': time.time() - start_time})
                break

            if token:
                # 发送 token 数据
                yield _sse({'type': 'token', 'token': token, 'session_id': session_id})

            if is_complete:
                # 发送结束事件
                yield _sse({'type': 'end', 'session_id': session_id, 'is_complete': True,
                           'processing_time': time.time() - start_time})
                break

            await asyncio.sleep(0.01)  # 控制流式输出的速度

    except Exception as e:
        # 发送错误事件
        yield _sse({'type': 'error', 'error': str(e), 'session_id': session_id})
        print(f"SSE stream error: {str(e)}")


# 流式查询 SSE 接口
@app.post("/api/stream")
async def stream_query(request: QueryRequest):
    """Server-Sent Events 流式查询接口"""
    # 使用请求中的 session_id 或生成新 ID
    session_id = request.session_id or str(uuid.uuid4())

    # 返回 SSE 流式响应
    return StreamingResponse(
        generate_sse_stream(request.query, request.source_filter, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用 Nginx 缓冲
        }
    )

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# 获取有效的学科类别
@app.get("/api/sources")
async def get_sources():
    return {"sources": qa_system.config.VALID_SOURCES}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=False)
