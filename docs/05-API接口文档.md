# API 接口文档

Web 服务由 `app.py`（FastAPI）提供，监听 `0.0.0.0:8080`。启动方式：

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

## 1. 请求/响应模型

```json
// QueryRequest
{ "query": "string", "source_filter": "string | null", "session_id": "string | null" }

// QueryResponse
{ "answer": "string", "is_streaming": "boolean", "session_id": "string", "processing_time": "number" }
```

## 2. 接口清单

### GET `/`
返回静态前端页面 `static/index.html`。

### POST `/api/create_session`
创建新会话。
- 响应：`{ "session_id": "uuid" }`

### POST `/api/query`
**非流式**查询。
- 请求体：`QueryRequest`
- 逻辑：
  1. 命中问候模板（你好/你是谁/在吗/干嘛呢）→ 直接返回模板回复（`is_streaming=false`）
  2. BM25 命中 → 返回精确答案（`is_streaming=false`）
  3. 未命中（需 RAG）→ 返回 `{ "answer": "请使用WebSocket接口获取流式响应", "is_streaming": true }`
- 响应：`QueryResponse`

### POST `/api/stream`
**SSE 流式**查询（RAG 走此接口）。
- 请求体：`QueryRequest`
- 响应：`text/event-stream`，事件类型：
  - `start`：`{ "type":"start", "session_id" }`
  - `token`：`{ "type":"token", "token":"...", "session_id" }`
  - `end`：`{ "type":"end", "session_id", "is_complete":true, "processing_time" }`
  - `error`：`{ "type":"error", "error":"..." }`

前端可通过 `EventSource` 或 `fetch` + `ReadableStream` 消费。

### GET `/api/history/{session_id}`
获取会话历史。
- 响应：`{ "session_id": "...", "history": [ { "question":"...", "answer":"..." } ] }`

### DELETE `/api/history/{session_id}`
清除会话历史。
- 响应：`{ "status":"success", "message":"历史记录已清除" }`

### GET `/api/sources`
获取可用学科类别。
- 响应：`{ "sources": ["ai","java","test","ops","bigdata"] }`

### GET `/health`
健康检查。
- 响应：`{ "status":"healthy" }`

## 3. 调用示例

```bash
# 创建会话
SESSION=$(curl -s -X POST http://localhost:8080/api/create_session | python -c "import sys,json;print(json.load(sys.stdin)['session_id'])")

# 非流式：精确问题
curl -s -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"windows如何安装redis\",\"session_id\":\"$SESSION\"}"

# 流式：专业咨询（SSE）
curl -N -X POST http://localhost:8080/api/stream \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"AI学科学费多少？\",\"session_id\":\"$SESSION\"}"
```

## 4. 前端页面

`static/index.html`（含 `static/src` 前端资源）为内置问答 UI，根路径 `/` 直接返回。压测脚本见 `locust_test.py`（覆盖 `/api/query` 与 WebSocket 场景，结果写入 `token_response_times.csv`）。
