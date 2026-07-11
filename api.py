from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Iterator

from fastapi import FastAPI, UploadFile, Form, File, Depends, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    RemoveMessage,
)
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

from gateway import chat, CALL_LOGS
import backend as be
from backend import (
    chatbot,
    ingest_rag_document,
    add_user_memory_fact,
    extract_memory_fact,
    save_thread_title,
    delete_thread,
    is_thread_empty,
    guard_input,
    guard_output,
    get_all_threads,
    GuardrailViolation,
)
from auth import verify_google_token, issue_session, user_from_session
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HERE = os.path.dirname(os.path.abspath(__file__))

# Runs carrying this tag (background summariser, memory extractor, doc
# summariser) must never surface to the user. With stream_mode="messages" every
# LLM call inside a node streams its tokens, so we filter tagged runs out below.
NO_STREAM_TAG = "no_stream"


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _pending_interrupts(state: Any) -> list[Any]:
    pending = list(getattr(state, "interrupts", None) or [])
    if not pending:
        for task in getattr(state, "tasks", None) or []:
            pending.extend(getattr(task, "interrupts", None) or [])
    return pending


def _owns(thread_id: str, user_id: str) -> bool:
    return thread_id.startswith(f"{user_id}::")


def _trim_last_ai_turn(thread_id: str) -> bool:
    """Remove the trailing assistant/tool messages back to (but not including)
    the last human message, so the graph can regenerate a fresh answer instead
    of appending a duplicate turn."""
    config = _config(thread_id)
    try:
        msgs = chatbot.get_state(config).values.get("messages", [])
    except Exception:
        return False
    remove_ids: list[str] = []
    for message in reversed(msgs):
        if isinstance(message, HumanMessage):
            break
        mid = getattr(message, "id", None)
        if mid:
            remove_ids.append(mid)
    if not remove_ids:
        return False
    chatbot.update_state(
        config, {"messages": [RemoveMessage(id=i) for i in remove_ids]}
    )
    return True


class GoogleLogin(BaseModel):
    credential: str


class DevLogin(BaseModel):
    email: str


class RenameBody(BaseModel):
    title: str


class ChatBody(BaseModel):
    thread_id: str
    message: str | None = None
    resume: str | None = None
    mode: str | None = None
    regenerate: bool = False


MODE_HINTS: dict[str, str] = {
    "document": "[Use the uploaded document via rag_tool to answer this.]",
    "calculate": "[Use the calculator tool for this.]",
    "web": "[Use web search for current information on this.]",
}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(HERE, "runova.html"))


@app.get("/api/config")
def config():
    return {"google_client_id": os.getenv("GOOGLE_CLIENT_ID", "")}


@app.get("/favicon.png", include_in_schema=False)
def favicon():
    return FileResponse("favicon.png")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@app.post("/api/auth/google")
def auth_google(body: GoogleLogin) -> dict[str, Any]:
    claims = verify_google_token(body.credential)
    return {
        "session": issue_session(claims),
        "user": {
            "id": f"google:{claims['sub']}",
            "email": claims.get("email", ""),
            "name": claims.get("name", claims.get("email", "User")),
            "picture": claims.get("picture", ""),
        },
    }


@app.post("/api/auth/dev")
def auth_dev(body: DevLogin) -> dict[str, Any]:
    if os.getenv("ALLOW_DEV_LOGIN") != "1":
        raise HTTPException(403, "Dev login disabled.")
    name = body.email.split("@")[0]
    claims = {
        "sub": body.email,
        "email": body.email,
        "name": name,
        "email_verified": True,
    }
    return {
        "session": issue_session(claims),
        "user": {"id": f"google:{body.email}", "email": body.email, "name": name},
    }


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------
@app.get("/api/threads")
def threads(user: dict = Depends(user_from_session)) -> dict[str, Any]:
    uid = user["sub"]
    everything = get_all_threads()
    return {"threads": {t: n for t, n in everything.items() if _owns(t, uid)}}


@app.patch("/api/threads/{thread_id}")
def rename_thread(
    thread_id: str,
    body: RenameBody,
    user: dict = Depends(user_from_session),
) -> dict[str, bool]:
    if not _owns(thread_id, user["sub"]):
        raise HTTPException(403, "Not your thread.")
    save_thread_title(thread_id, body.title.strip() or "New Chat")
    return {"ok": True}


@app.delete("/api/threads/{thread_id}")
def remove_thread(
    thread_id: str, user: dict = Depends(user_from_session)
) -> dict[str, bool]:
    if not _owns(thread_id, user["sub"]):
        raise HTTPException(403, "Not your thread.")
    delete_thread(thread_id)
    return {"ok": True}


@app.get("/api/threads/{thread_id}/messages")
def thread_messages(
    thread_id: str, user: dict = Depends(user_from_session)
) -> dict[str, Any]:
    if not _owns(thread_id, user["sub"]):
        raise HTTPException(403, "Not your thread.")
    try:
        state = chatbot.get_state(_config(thread_id))
        stored = state.values.get("messages", [])
    except Exception:
        stored = []
    out: list[dict[str, Any]] = []
    for message in stored:
        if isinstance(message, HumanMessage):
            out.append({"role": "user", "content": _text(message.content)})
        elif isinstance(message, AIMessage) and _text(message.content).strip():
            out.append(
                {"role": "bot", "content": _text(message.content), "flags": []}
            )
    return {"messages": out}


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
@app.post("/api/upload")
def upload(
    file: UploadFile = File(...),
    thread_id: str = Form(...),
    user: dict = Depends(user_from_session),
) -> JSONResponse:
    if not _owns(thread_id, user["sub"]):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    suffix = os.path.splitext(file.filename or "")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        path = tmp.name
    try:
        count = ingest_rag_document(path, thread_id)
        return JSONResponse({"chunks": count})
    except Exception as error:
        import traceback; traceback.print_exc() 
        return JSONResponse({"error": str(error)}, status_code=400)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Memory (scoped to the verified user via namespaced source_thread_id)
# ---------------------------------------------------------------------------
@app.get("/api/memory")
def memory(user: dict = Depends(user_from_session)) -> dict[str, Any]:
    cur = be.conn.execute(
        "SELECT id, fact FROM user_memory "
        "WHERE source_thread_id LIKE ? ORDER BY id DESC LIMIT ?",
        (f"{user['sub']}::%", be.MAX_MEMORY_FACTS),
    )
    return {"facts": [(int(r[0]), str(r[1])) for r in cur.fetchall()]}


@app.delete("/api/memory/{fact_id}")
def memory_delete(
    fact_id: int, user: dict = Depends(user_from_session)
) -> dict[str, bool]:
    cur = be.conn.execute(
        "SELECT source_thread_id FROM user_memory WHERE id = ?", (fact_id,)
    )
    row = cur.fetchone()
    if not row or not (row[0] or "").startswith(f"{user['sub']}::"):
        raise HTTPException(403, "Not your memory.")
    be.conn.execute("DELETE FROM user_memory WHERE id = ?", (fact_id,))
    be.conn.commit()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Chat stream
# ---------------------------------------------------------------------------
def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _is_background(metadata: dict[str, Any]) -> bool:
    """True if this streamed chunk belongs to a tagged background run (summary,
    memory extraction, doc summary) that must not reach the user."""
    return NO_STREAM_TAG in (metadata.get("tags") or [])


def _maybe_title(thread_id: str, first_msg: str) -> None:
    if not is_thread_empty(thread_id):
        return
    try:
        title = (
            chat(
                "Write a 3-6 word title for a chat that starts with this "
                "message. Reply with only the title, no quotes.\n\n" + first_msg,
                tier="aux",
                guard=False,
                tags=[NO_STREAM_TAG],
            )
            .strip()
            .strip('"')
        )
    except Exception:
        title = "New Chat"
    save_thread_title(thread_id, title or "New Chat")


def _count_turn_sources(state: Any) -> int:
    """Count retrieval/search tool results produced since the last human turn."""
    turn_tools = 0
    for message in reversed(state.values.get("messages", [])):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage) and getattr(
            message, "name", ""
        ) in ("rag_tool", "search_tool"):
            turn_tools += 1
    return turn_tools


def _stream(body: ChatBody, user_id: str) -> Iterator[str]:
    if not _owns(body.thread_id, user_id):
        yield _sse({"type": "error", "message": "forbidden"})
        return

    config = _config(body.thread_id)

    if body.regenerate:
        if not _trim_last_ai_turn(body.thread_id):
            yield _sse({"type": "error", "message": "Nothing to regenerate."})
            return
        # Empty update reruns chat_node on the existing (untouched) human turn.
        payload: Any = {"messages": []}
        user_text: str | None = None
    elif body.resume is not None:
        payload = Command(resume=body.resume)
        user_text = None
    else:
        try:
            clean = guard_input(body.message or "")
        except GuardrailViolation as violation:
            yield _sse({"type": "token", "text": str(violation)})
            yield _sse({"type": "flags", "flags": ["blocked"]})
            yield _sse({"type": "done"})
            return
        _maybe_title(body.thread_id, clean)
        hint = MODE_HINTS.get(body.mode or "", "")
        graph_text = f"{hint}\n\n{clean}" if hint else clean
        payload = {"messages": [HumanMessage(content=graph_text)]}
        user_text = clean

    final_text = ""
    try:
        for chunk in chatbot.stream(
            payload, config=config, stream_mode="messages"
        ):
            message, metadata = chunk if isinstance(chunk, tuple) else (chunk, {})
            if metadata.get("langgraph_node") != "chat_node":
                continue
            # Drop tokens from tagged background runs (summariser, etc.) so they
            # never leak into the user-facing answer.
            if _is_background(metadata):
                continue
            if isinstance(message, AIMessage):
                piece = _text(message.content)
                if piece:
                    final_text += piece
                    yield _sse({"type": "token", "text": piece})
    except Exception as error:
        # RunnableWithFallbacks re-raises only the FIRST exception in the
        # chain, so `error` alone can be misleading if a later fallback model
        # also failed for a different reason. Pull the tail of CALL_LOGS
        # (populated by GatewayLogger.log_failure_event for every attempt,
        # success or failure) so we can see every model that was actually
        # tried during this turn, in order.
        recent_failures = [
            entry for entry in CALL_LOGS[-6:] if "error" in entry
        ]
        if recent_failures:
            print(f"[chat:{body.thread_id}] fallback chain attempts:")
            for entry in recent_failures:
                print(f"    {entry.get('model')}: {entry.get('error')}")
        yield _sse({"type": "error", "message": str(error)})
        if recent_failures:
            yield _sse({"type": "fallback_trace", "attempts": recent_failures})
        return

    state = chatbot.get_state(config)
    pending = _pending_interrupts(state)
    if pending:
        value = getattr(pending[0], "value", pending[0])
        yield _sse({"type": "interrupt", "value": str(value)})
        return

    if not final_text.strip():
        stored = state.values.get("messages", [])
        for message in reversed(stored):
            if isinstance(message, AIMessage) and _text(message.content).strip():
                final_text = _text(message.content)
                break

    safe, flags = guard_output(final_text)
    if safe != final_text:
        yield _sse(
            {"type": "token", "text": "\n\n[response adjusted by guardrail]"}
        )
    if flags:
        yield _sse({"type": "flags", "flags": flags})

    sources = _count_turn_sources(state)
    if sources:
        yield _sse({"type": "meta", "sources": sources})

    if user_text and safe:
        fact = extract_memory_fact(user_text, safe)
        if fact:
            add_user_memory_fact(fact, body.thread_id)

    yield _sse({"type": "done"})


@app.post("/api/chat")
def chat_stream(
    body: ChatBody, user: dict = Depends(user_from_session)
) -> StreamingResponse:
    return StreamingResponse(
        _stream(body, user["sub"]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )