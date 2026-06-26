"""
YiAgent — Enterprise-Grade AI Agent Runtime

Startup:  python app.py
"""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from yiagent.common.config import load_config, conf
from yiagent.common.log import logger
from yiagent.protocol.models import AgentEvent, ContentBlock, Message


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_config()
    logger.info("[YiAgent] Starting up")

    # ---- BUG-1 fix: register model provider ----
    from yiagent.model_gateway.router import register_provider
    try:
        from yiagent.model_gateway.providers.openai import OpenAIProvider
        register_provider("openai", OpenAIProvider)
        logger.info("[YiAgent] OpenAI provider registered")
    except Exception as e:
        logger.warning(f"[YiAgent] Model provider registration failed: {e}")

    # ---- BUG-2/3 fix: graceful degradation for infra ----
    app.state.redis_ok = False
    try:
        from yiagent.common.redis_pool import get_redis
        await get_redis()
        app.state.redis_ok = True
        logger.info("[YiAgent] Redis connected")
    except Exception as e:
        logger.warning(f"[YiAgent] Redis not available: {e}")

    app.state.pg_ok = False
    try:
        from yiagent.memory.long_term_store import get_long_term_store
        store = get_long_term_store()
        await store.initialize()
        app.state.pg_ok = True
        logger.info("[YiAgent] PostgreSQL connected")
    except Exception as e:
        logger.warning(f"[YiAgent] PostgreSQL not available: {e}")

    from yiagent.channel.adapters.web import WebChannelAdapter
    from yiagent.channel.base import register_adapter
    app.state.web_adapter = WebChannelAdapter()
    register_adapter("web", app.state.web_adapter)
    logger.info("[YiAgent] Web channel registered")

    yield

    logger.info("[YiAgent] Shutting down")
    if app.state.redis_ok:
        from yiagent.common.redis_pool import close_redis
        try:
            await asyncio.wait_for(close_redis(), timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            pass


app = FastAPI(title="YiAgent", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Session store with TTL-based expiry (BUG-6 fix)
# ---------------------------------------------------------------------------

_SESSION_TTL = 3600  # 1 hour
_sessions: dict = {}


def _purge_sessions():
    """Drop sessions older than TTL."""
    now = time.time()
    stale = [sid for sid, (_, ts) in _sessions.items() if now - ts > _SESSION_TTL]
    for sid in stale:
        _sessions.pop(sid, None)
    if stale:
        logger.debug(f"[API] Purged {len(stale)} expired sessions")


# ---------------------------------------------------------------------------
# Helper: build agent for a session
# ---------------------------------------------------------------------------

async def _get_or_create_agent(session_id: str):
    from yiagent.agent.runtime import AgentRuntime
    from yiagent.agent.tools.tool_manager import ToolManager
    from yiagent.memory.manager import MemoryManager
    from yiagent.memory.conversation_store import get_conversation_store
    from yiagent.model_gateway.router import create_model_router

    router = create_model_router()
    model = router.primary
    if model is None:
        raise HTTPException(status_code=503, detail="No model provider configured. Set YIAGENT_OPENAI_API_KEY.")

    model.session_id = session_id
    model.channel_type = "web"

    tm = ToolManager()
    # Register built-in tools
    from yiagent.agent.tools.builtin.memory_tools import (
        MemorySearchTool, MemoryGetTool, MemoryAddTool,
    )
    tm.register_tool(MemorySearchTool)
    tm.register_tool(MemoryGetTool)
    tm.register_tool(MemoryAddTool)
    tm.load_mcp_tools()
    tools = tm.create_all_instances()

    memory_manager = MemoryManager()
    if app.state.pg_ok:
        try:
            await memory_manager.initialize()
        except Exception as e:
            logger.warning(f"[API] MemoryManager init failed (PG down?): {e}")

    cfg = conf()
    ws = os.path.expanduser(cfg.get("workspace_root", "~/yiagent"))
    skills_dir = os.path.join(ws, "skills")

    try:
        from yiagent.agent.skills.manager import SkillManager
        skill_mgr = SkillManager(custom_dir=skills_dir)
    except Exception:
        skill_mgr = None

    runtime_info = {"workspace": ws, "model": model.model, "channel": "web"}

    store = await get_conversation_store()

    agent = AgentRuntime(
        session_id=session_id,
        model=model,
        tools=tools,
        max_steps=cfg.get("agent_max_steps", 50),
        max_context_turns=cfg.get("agent_max_context_turns", 30),
        max_context_tokens=cfg.get("agent_max_context_tokens", 50000),
        workspace_dir=ws,
        memory_manager=memory_manager,
        skill_manager=skill_mgr,
        runtime_info=runtime_info,
    )

    if app.state.redis_ok:
        try:
            agent.messages = await store.load_context(session_id, max_turns=agent.max_context_turns)
        except Exception as e:
            logger.warning(f"[API] Context load failed: {e}")

    return agent


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "redis": app.state.redis_ok,
        "postgres": app.state.pg_ok,
        "sessions": len(_sessions),
    }


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    session_id = body.get("session_id", f"web_{uuid.uuid4().hex[:12]}")
    user_text = body.get("content", "")
    user_id = body.get("user_id", "")

    if not user_text:
        raise HTTPException(status_code=400, detail="content is required")

    _purge_sessions()
    cached = _sessions.get(session_id)
    if cached:
        agent, _ = cached
    else:
        agent = await _get_or_create_agent(session_id)
    _sessions[session_id] = (agent, time.time())

    msg = Message(
        role="user",
        content=[ContentBlock.text_block(user_text)],
        session_id=session_id,
        user_id=user_id,
        channel_type="web",
    )

    try:
        response = await agent.handle_message(msg)
    except Exception as e:
        logger.error(f"[API] Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"session_id": session_id, "response": response}


@app.get("/chat/stream")
async def chat_stream(request: Request):
    session_id = request.query_params.get("session_id", f"web_{uuid.uuid4().hex[:12]}")
    user_text = request.query_params.get("content", "")
    user_id = request.query_params.get("user_id", "")

    if not user_text:
        raise HTTPException(status_code=400, detail="content is required")

    _purge_sessions()
    cached = _sessions.get(session_id)
    if cached:
        agent, _ = cached
    else:
        agent = await _get_or_create_agent(session_id)
    _sessions[session_id] = (agent, time.time())

    msg = Message(
        role="user",
        content=[ContentBlock.text_block(user_text)],
        session_id=session_id,
        user_id=user_id,
        channel_type="web",
    )

    queue: asyncio.Queue = asyncio.Queue(maxsize=256)

    def on_event(event: AgentEvent):
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    async def event_generator():
        from yiagent.memory.conversation_store import _bg_tracker
        task = asyncio.create_task(agent.handle_message(msg, on_event=on_event))
        task.add_done_callback(_bg_tracker._log_failure)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                if event.type == "agent_end":
                    data = json.dumps({"type": "done", "content": event.data.get("final_response", "")},
                                      ensure_ascii=False)
                    yield f"event: done\ndata: {data}\n\n"
                    break

                if event.type in ("message_update", "reasoning_update"):
                    delta = event.data.get("delta", "")
                    data = json.dumps({"type": event.type, "delta": delta}, ensure_ascii=False)
                    yield f"event: {event.type}\ndata: {data}\n\n"

                elif event.type == "tool_execution_end":
                    data = json.dumps({
                        "type": "tool_end",
                        "tool": event.data.get("tool_name", ""),
                        "status": event.data.get("status", ""),
                    }, ensure_ascii=False)
                    yield f"event: tool_end\ndata: {data}\n\n"

                elif event.type == "error":
                    data = json.dumps({"type": "agent_error", "message": event.data.get("error", "")},
                                      ensure_ascii=False)
                    yield f"event: agent_error\ndata: {data}\n\n"
                    break

            await task
        except asyncio.CancelledError:
            task.cancel()
        except Exception as e:
            logger.error(f"[SSE] Stream error: {e!r}", exc_info=True)
            data = json.dumps({"type": "agent_error", "message": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
            yield f"event: agent_error\ndata: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id,
        },
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_CHAT_HTML)


_CHAT_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YiAgent Chat</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
background:#1a1a2e;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
header{background:#16213e;padding:12px 20px;font-size:18px;font-weight:600;
border-bottom:1px solid #0f3460}
#messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:12px}
.msg{max-width:80%;padding:12px 16px;border-radius:12px;line-height:1.5;white-space:pre-wrap}
.msg.user{align-self:flex-end;background:#0f3460}
.msg.assistant{align-self:flex-start;background:#16213e}
.tool{font-size:12px;color:#888;margin:4px 0}
#input-area{display:flex;padding:12px 20px;gap:8px;border-top:1px solid #0f3460;
background:#16213e}
#input{flex:1;padding:10px 14px;border-radius:8px;border:1px solid #0f3460;
background:#1a1a2e;color:#e0e0e0;font-size:14px;resize:none;outline:none}
#send{padding:10px 24px;border-radius:8px;border:none;background:#e94560;color:#fff;
font-size:14px;cursor:pointer}
#send:hover{opacity:.9}
#status{font-size:12px;color:#888;text-align:center;padding:4px}
</style>
</head>
<body>
<header>YiAgent Chat</header>
<div id="messages"></div>
<div id="status">&nbsp;</div>
<div id="input-area">
<textarea id="input" rows="2" placeholder="输入消息…" onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea>
<button id="send" onclick="send()">发送</button>
</div>
<script>
const sessionId = "web_" + Math.random().toString(36).slice(2,14);
let currentMsg = null;

function addMsg(role, text) {
    const div = document.createElement("div");
    div.className = "msg " + role;
    div.textContent = text;
    document.getElementById("messages").appendChild(div);
    document.getElementById("messages").scrollTop = document.getElementById("messages").scrollHeight;
    return div;
}

async function send() {
    const input = document.getElementById("input");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    document.getElementById("status").textContent = "thinking…";

    addMsg("user", text);
    currentMsg = addMsg("assistant", "");
    const sendBtn = document.getElementById("send");
    sendBtn.disabled = true;

    const url = `/chat/stream?session_id=${encodeURIComponent(sessionId)}&content=${encodeURIComponent(text)}`;
    const es = new EventSource(url);

    es.addEventListener("message_update", e => {
        const d = JSON.parse(e.data);
        currentMsg.textContent += d.delta;
        document.getElementById("messages").scrollTop = document.getElementById("messages").scrollHeight;
    });

    es.addEventListener("tool_end", e => {
        // Show subtle activity indicator only — don't expose internal tool status to user
        document.getElementById("status").textContent = "working…";
    });

    es.addEventListener("done", e => {
        es.close();
        sendBtn.disabled = false;
        document.getElementById("status").textContent = "ready";
    });

    es.addEventListener("agent_error", e => {
        let msg = "error";
        try { const d = JSON.parse(e.data); msg = d.message || msg; } catch(_){}
        currentMsg.textContent += "\\n[Error: " + msg + "]";
        es.close();
        sendBtn.disabled = false;
        document.getElementById("status").textContent = "error";
    });

    es.onerror = () => {
        if (es.readyState === EventSource.CLOSED) return;
        es.close();
        sendBtn.disabled = false;
        document.getElementById("status").textContent = "connection lost";
    };
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
