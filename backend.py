from __future__ import annotations
import re
import ast
import math
import operator
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt
from typing import TypedDict, Annotated

from gateway import (
    get_llm,
    chat,
    guard_input,
    guard_output,
    get_usage_report,
    reset_usage,
    GuardrailViolation,
)

__all__ = [
    "chatbot",
    "embeddings",
    "get_retriever",
    "retrieve_context",
    "ingest_rag_document",
    "get_document_loader",
    "get_user_memory",
    "get_user_memory_with_ids",
    "get_relevant_user_memory",
    "add_user_memory_fact",
    "delete_user_memory_fact",
    "clear_user_memory",
    "extract_memory_fact",
    "get_all_threads",
    "save_thread_title",
    "delete_thread",
    "is_thread_empty",
    "tools",
    "guard_input",
    "guard_output",
    "get_usage_report",
    "reset_usage",
    "GuardrailViolation",
]

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatbot.db")
FAISS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_db")

RAG_FETCH_K = 20
RAG_TOP_K = 3
RAG_MMR_LAMBDA = 0.5
RAG_SCORE_THRESHOLD = 0.35
RAG_RERANK_ENABLED = os.getenv("RAG_RERANK", "0") == "1"
RAG_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

_reranker: Any = None


def _get_reranker() -> Any:
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

        _reranker = CrossEncoder(RAG_RERANK_MODEL)
    return _reranker


def _faiss_path(thread_id: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*]', "_", thread_id)
    return os.path.join(FAISS_ROOT, safe)

class SimpleImageOCRLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
    def load(self):
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(Image.open(self.file_path))
        return [Document(page_content=text, metadata={"source": self.file_path})]

def get_document_loader(file_path: str) -> Any:
    extension = Path(file_path).suffix.lower()
    if extension == ".pdf":
        return PyPDFLoader(file_path)
    if extension == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    if extension == ".csv":
        return CSVLoader(file_path)
    if extension == ".docx":
        return Docx2txtLoader(file_path)
    if extension in [".ppt", ".pptx"]:
        return UnstructuredPowerPointLoader(file_path)
    if extension == ".md":
        return UnstructuredMarkdownLoader(file_path)
    if extension in [".html", ".htm"]:
        return UnstructuredHTMLLoader(file_path)
    if extension in [".xlsx", ".xls"]:
        return UnstructuredExcelLoader(file_path, mode="elements")
    if extension in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"]:
        return SimpleImageOCRLoader(file_path)
    raise ValueError(f"Unsupported file type: {extension}")


def ingest_rag_document(file_path: str, thread_id: str) -> int:
    db_path = _faiss_path(thread_id)
    os.makedirs(db_path, exist_ok=True)
    loader = get_document_loader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    index_file = os.path.join(db_path, "index.faiss")
    if os.path.exists(index_file):
        vector_store = FAISS.load_local(
            db_path, embeddings, allow_dangerous_deserialization=True
        )
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(db_path)
    return len(chunks)


def _load_vector_store(thread_id: str) -> Any | None:
    db_path = _faiss_path(thread_id)
    if not os.path.exists(os.path.join(db_path, "index.faiss")):
        return None
    return FAISS.load_local(
        db_path, embeddings, allow_dangerous_deserialization=True
    )


def get_retriever(thread_id: str) -> Any | None:
    vector_store = _load_vector_store(thread_id)
    if vector_store is None:
        return None
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RAG_TOP_K,
            "fetch_k": RAG_FETCH_K,
            "lambda_mult": RAG_MMR_LAMBDA,
        },
    )


def retrieve_context(thread_id: str, query: str) -> list[Document]:
    vector_store = _load_vector_store(thread_id)
    if vector_store is None:
        return []

    scored = vector_store.max_marginal_relevance_search_with_score_by_vector(
        embeddings.embed_query(query),
        k=RAG_TOP_K,
        fetch_k=RAG_FETCH_K,
        lambda_mult=RAG_MMR_LAMBDA,
    ) if hasattr(
        vector_store, "max_marginal_relevance_search_with_score_by_vector"
    ) else [
        (doc, 0.0)
        for doc in vector_store.max_marginal_relevance_search(
            query, k=RAG_TOP_K, fetch_k=RAG_FETCH_K, lambda_mult=RAG_MMR_LAMBDA
        )
    ]

    candidates = [doc for doc, score in scored]

    if not candidates:
        return []

    if RAG_RERANK_ENABLED and candidates:
        reranker = _get_reranker()
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(
            zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True
        )
        candidates = [
            doc for doc, score in ranked if float(score) >= RAG_SCORE_THRESHOLD
        ] or [ranked[0][0]]

    return candidates[:RAG_TOP_K]


def _format_documents(documents: list[Document]) -> str:
    formatted: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "N/A")
        formatted.append(
            f"[{index}] source={source} page={page}\n{document.page_content}"
        )
    return "\n\n".join(formatted)


@tool
def rag_tool(query: str, config: RunnableConfig) -> str:
    """Search the uploaded document for this conversation. Call this FIRST for any
    question that could be answered by an uploaded PDF, notes, or file, and only
    say no document is available if this returns 'No relevant information'.

    Args:
        query: The question or search query used to retrieve document content.
    """
    configurable = config.get("configurable", {}) if config else {}
    thread_id = configurable.get("thread_id")
    if not thread_id:
        return "No relevant information was found."

    if _load_vector_store(thread_id) is None:
        return "No document has been uploaded for this conversation."

    documents = retrieve_context(thread_id, query)
    if not documents:
        return "No relevant information was found."

    return _format_documents(documents)


search_tool = TavilySearch(max_results=3, topic="general", search_depth="advanced")

_ALLOWED_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
}

_ALLOWED_NAMES: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise ValueError("Operator not allowed.")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_UNARY.get(type(node.op))
        if op is None:
            raise ValueError("Unary operator not allowed.")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"Name not allowed: {node.id}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed.")
        func = _ALLOWED_FUNCS.get(node.func.id)
        if func is None:
            raise ValueError(f"Function not allowed: {node.func.id}")
        args = [_eval_node(arg) for arg in node.args]
        return func(*args)
    raise ValueError("Expression element not allowed.")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression safely. Supports + - * / // % **, parentheses,
    and functions like sqrt, log, sin, cos, abs, round, min, max, pow.
    Example: '2 + 2', 'sqrt(16)', '10 * 5'.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree))
    except Exception as error:
        return f"Calculation error: {error}"


@tool
def get_stock_price(symbol: str) -> dict[str, Any]:
    """Fetch the latest stock quote for a symbol (e.g. 'AAPL', 'TSLA') via Alpha Vantage."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {"error": "ALPHAVANTAGE_API_KEY is not set."}
    url = (
        f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        return {"error": f"Could not fetch stock price: {error}"}


@tool
def purchase_stock(symbol: str, quantity: int) -> dict[str, Any]:
    """Simulate purchasing a quantity of a stock symbol. Interrupts for human
    approval before confirming the order."""
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")
    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    return {
        "status": "cancelled",
        "message": f"Purchase of {quantity} shares of {symbol} was declined.",
        "symbol": symbol,
        "quantity": quantity,
    }


@tool
def get_current_datetime(timezone_name: str = "UTC") -> str:
    """Return the current date and time. Use for any 'what time/date is it',
    'today', 'now', or scheduling question.

    Args:
        timezone_name: An IANA timezone name like 'Asia/Kolkata' or 'US/Eastern'. Defaults to UTC.
    """
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(timezone_name) if timezone_name.upper() != "UTC" else timezone.utc
    except Exception:
        tz = timezone.utc
        timezone_name = "UTC"
    now = datetime.now(tz)
    return now.strftime(f"%A, %d %B %Y, %H:%M:%S ({timezone_name})")


@tool
def summarize_document(config: RunnableConfig, focus: str = "") -> str:
    """Summarize the document uploaded for this conversation. Optionally focus the
    summary on a specific topic.

    Args:
        focus: Optional topic to focus the summary on. Leave empty for a general summary.
    """
    configurable = config.get("configurable", {}) if config else {}
    thread_id = configurable.get("thread_id")
    if not thread_id or _load_vector_store(thread_id) is None:
        return "No document has been uploaded for this conversation."

    query = focus or "overall summary key points main topics conclusions"
    documents = retrieve_context(thread_id, query)
    if not documents:
        return "No relevant information was found in the document."

    context = _format_documents(documents)
    instruction = (
        "Summarize the following document excerpts into a clear, faithful summary. "
        "Use only the provided content and do not invent details."
    )
    if focus:
        instruction += f" Focus on: {focus}."
    answer = chat(
        f"{instruction}\n\nExcerpts:\n{context}",
        tier="aux",
        guard=False,
        tags=["no_stream"],
    )
    safe, _ = guard_output(answer, context=context)
    return safe


@tool
def get_current_weather(location: str) -> str:
    """Get the current real-time weather for a city or location.

    Args:
        location: City or location name, e.g. 'Dhaka', 'London, UK', or 'New York, US'.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather API key is missing. Set the OPENWEATHER_API_KEY environment variable."
    try:
        geocoding_params: dict[str, str | int] = {
            "q": location,
            "limit": 1,
            "appid": api_key,
        }
        geo_response = requests.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params=geocoding_params,
            timeout=10,
        )
        geo_response.raise_for_status()
        locations: list[dict[str, Any]] = geo_response.json()
        if not locations:
            return f"Could not find the location: {location}"

        latitude = locations[0]["lat"]
        longitude = locations[0]["lon"]
        resolved_name = locations[0].get("name", location)
        country = locations[0].get("country", "")
        state = locations[0].get("state", "")

        weather_params: dict[str, str | float] = {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric",
        }
        weather_response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params=weather_params,
            timeout=10,
        )
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        pressure = weather_data["main"]["pressure"]
        description = weather_data["weather"][0]["description"]
        wind_speed = weather_data.get("wind", {}).get("speed", "N/A")
        visibility_meters = weather_data.get("visibility")
        visibility_km = (
            round(visibility_meters / 1000, 1)
            if visibility_meters is not None
            else "N/A"
        )

        location_parts = [resolved_name]
        if state:
            location_parts.append(state)
        if country:
            location_parts.append(country)
        display_location = ", ".join(location_parts)

        return (
            f"Current weather in {display_location}:\n"
            f"- Condition: {description.title()}\n"
            f"- Temperature: {temperature}°C\n"
            f"- Feels like: {feels_like}°C\n"
            f"- Humidity: {humidity}%\n"
            f"- Pressure: {pressure} hPa\n"
            f"- Wind speed: {wind_speed} m/s\n"
            f"- Visibility: {visibility_km} km"
        )
    except requests.Timeout:
        return "The weather service request timed out. Please try again."
    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response else "unknown"
        if status_code == 401:
            return "The OpenWeather API key is invalid or inactive."
        return f"Weather API returned an HTTP error: {status_code}"
    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"
    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected weather API response: {error}"


tools: list[Any] = [
    search_tool,
    calculator,
    get_stock_price,
    get_current_weather,
    rag_tool,
    summarize_document,
    get_current_datetime,
    purchase_stock,
]

llm_with_tools = get_llm("balanced", tools=tools)

conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
checkpoint = SqliteSaver(conn)

conn.execute(
    "CREATE TABLE IF NOT EXISTS thread_titles ("
    "thread_id TEXT PRIMARY KEY, title TEXT)"
)
conn.execute(
    "CREATE TABLE IF NOT EXISTS user_memory ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "fact TEXT NOT NULL, "
    "source_thread_id TEXT, "
    "created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
)
conn.commit()

MAX_MEMORY_FACTS = 40
MEMORY_RECALL_K = 6
MEMORY_DEDUP_THRESHOLD = 0.92
MEMORY_RECALL_THRESHOLD = 0.30
SHORT_TERM_WINDOW_SIZE = 6


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def get_user_memory() -> list[str]:
    cur = conn.execute(
        "SELECT fact FROM user_memory ORDER BY id DESC LIMIT ?",
        (MAX_MEMORY_FACTS,),
    )
    return [row[0] for row in cur.fetchall()]


def get_user_memory_with_ids() -> list[tuple[int, str]]:
    cur = conn.execute(
        "SELECT id, fact FROM user_memory ORDER BY id DESC LIMIT ?",
        (MAX_MEMORY_FACTS,),
    )
    return [(int(row[0]), str(row[1])) for row in cur.fetchall()]


def get_relevant_user_memory(query: str, k: int = MEMORY_RECALL_K) -> list[str]:
    facts = get_user_memory()
    if not facts or not query.strip():
        return []
    query_vec = embeddings.embed_query(query)
    fact_vecs = embeddings.embed_documents(facts)
    scored = [
        (fact, _cosine(query_vec, vec)) for fact, vec in zip(facts, fact_vecs)
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    relevant = [fact for fact, score in scored if score >= MEMORY_RECALL_THRESHOLD]
    return relevant[:k]


def add_user_memory_fact(fact: str, thread_id: str | None = None) -> bool:
    fact = fact.strip()
    if not fact:
        return False
    existing = get_user_memory()
    if existing:
        new_vec = embeddings.embed_query(fact)
        for vec in embeddings.embed_documents(existing):
            if _cosine(new_vec, vec) >= MEMORY_DEDUP_THRESHOLD:
                return False
    conn.execute(
        "INSERT INTO user_memory (fact, source_thread_id) VALUES (?, ?)",
        (fact, thread_id),
    )
    conn.commit()
    return True


def delete_user_memory_fact(fact_id: int) -> None:
    conn.execute("DELETE FROM user_memory WHERE id = ?", (fact_id,))
    conn.commit()


def clear_user_memory() -> None:
    conn.execute("DELETE FROM user_memory")
    conn.commit()


def extract_memory_fact(user_message: str, ai_message: str) -> str | None:
    prompt = (
        "Below is one exchange from a conversation between a user and an "
        "AI assistant.\n\n"
        f"User: {user_message}\n"
        f"Assistant: {ai_message}\n\n"
        "Does this exchange reveal a durable fact worth remembering about "
        "the user for ALL of their future conversations, such as their name, "
        "role, a stated preference, or a recurring project/topic? Ignore "
        "one-off questions with no lasting relevance.\n\n"
        "If yes, reply with ONLY one short factual sentence (under 20 words), "
        'in third person, e.g. "Prefers concise answers without extra '
        'caveats." If no, reply with exactly: NONE'
    )
    try:
        # aux tier + no_stream tag: this runs as a post-turn call and must never
        # surface to the user.
        fact = chat(prompt, tier="aux", guard=False, tags=["no_stream"]).strip()
    except Exception:
        return None
    if not fact or fact.upper().strip(".") == "NONE":
        return None
    return fact


def summarize_messages(
    existing_summary: str, messages_to_fold: list[BaseMessage]
) -> str:
    if not messages_to_fold:
        return existing_summary
    convo_text = "\n".join(
        f"{message.type}: {content_to_text(message.content)}"
        for message in messages_to_fold
        if getattr(message, "content", None)
    )
    if not convo_text.strip():
        return existing_summary
    prompt = (
        "You are maintaining a running summary of an ongoing conversation "
        "between a user and an AI assistant, used to save context space.\n\n"
        f"Existing summary so far:\n{existing_summary or '(none yet)'}\n\n"
        f"New conversation turns to fold in:\n{convo_text}\n\n"
        "Write an updated, concise summary (roughly 100-150 words) that "
        "preserves important facts, decisions, numbers, and unresolved "
        "threads from BOTH the existing summary and the new turns. Reply "
        "with ONLY the updated summary text, nothing else."
    )
    try:
        # This call happens INSIDE chat_node. With stream_mode="messages" its
        # tokens would otherwise stream straight to the user, so it MUST carry
        # the no_stream tag (the API layer drops tagged runs). aux tier keeps it
        # off the shared Groq RPM budget.
        return (
            chat(prompt, tier="aux", guard=False, tags=["no_stream"]).strip()
            or existing_summary
        )
    except Exception:
        return existing_summary


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    summary_covers: int

def _sanitize_history(msg: BaseMessage) -> BaseMessage:
    if isinstance(msg, AIMessage):
        # strip reasoning (harmony won't accept it back)
        msg.additional_kwargs.pop("reasoning_content", None)
        if isinstance(msg.response_metadata, dict):
            msg.response_metadata.pop("reasoning_content", None)
        # drop any tool_call missing a name: harmony refuses to render these
        tcs = getattr(msg, "tool_calls", None)
        if tcs:
            good = [tc for tc in tcs if (tc.get("name") or "").strip()]
            if len(good) != len(tcs):
                msg.tool_calls = good
                # if we removed the calls, also clear the raw kwargs copy
                if not good:
                    msg.additional_kwargs.pop("tool_calls", None)
    return msg

def chat_node(state: ChatState) -> dict[str, Any]:
    all_messages = state["messages"]
    summary = state.get("summary", "") or ""
    summary_covers = state.get("summary_covers", 0) or 0

    older_boundary = max(0, len(all_messages) - SHORT_TERM_WINDOW_SIZE)
    if older_boundary > summary_covers:
        newly_expiring = all_messages[summary_covers:older_boundary]
        summary = summarize_messages(summary, newly_expiring)
        summary_covers = older_boundary

    recent_messages = (
        all_messages[-SHORT_TERM_WINDOW_SIZE:] if all_messages else []
    )

    recent_messages = [_sanitize_history(m) for m in recent_messages] 

    last_human = ""
    for message in reversed(all_messages):
        if isinstance(message, HumanMessage):
            last_human = content_to_text(message.content)
            break

    user_facts = get_relevant_user_memory(last_human)
    memory_block = ""
    if user_facts:
        memory_block = (
            "\n\nWhat you know about the user from past conversations "
            "(apply only when relevant):\n"
            + "\n".join(f"- {fact}" for fact in user_facts)
        )

    summary_block = ""
    if summary:
        summary_block = (
            "\n\nSummary of earlier parts of THIS conversation (older "
            f"messages have been condensed to save context space):\n{summary}"
        )

    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"
            "Tool usage instructions:\n"
            "- Use `rag_tool` for questions about the uploaded document. Always "
            "retrieve relevant content before answering document questions.\n"
            "- Use `summarize_document` when the user asks for a summary of the "
            "uploaded document.\n"
            "- Use `search_tool` for current events or information that needs an "
            "internet search.\n"
            "- Use `calculator` for mathematical calculations.\n"
            "- Use `get_stock_price` for current stock prices.\n"
            "- Use `get_current_datetime` for the current date/time.\n\n"
            "Answer general questions directly when no tool is required. Never "
            "invent information from the uploaded document. If the user asks "
            "about a document but none is available, ask them to upload one. "
            "When you use retrieved document content, cite the bracketed source "
            "markers. After a tool result, give a clear final answer."
            + memory_block
            + summary_block
        )
    )

    messages: list[BaseMessage] = [system_message, *recent_messages]
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "summary": summary,
        "summary_covers": summary_covers,
    }


tool_node = ToolNode(tools)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpoint)


def get_all_threads() -> dict[str, str]:
    thread_latest_ts: dict[str, str] = {}
    for ckpt in checkpoint.list(None):
        thread_id = ckpt.config["configurable"]["thread_id"]
        if thread_id.startswith("title-"):
            continue
        ts = ckpt.checkpoint.get("ts") or ""
        if thread_id not in thread_latest_ts or ts > thread_latest_ts[thread_id]:
            thread_latest_ts[thread_id] = ts
    if not thread_latest_ts:
        return {}
    sorted_thread_ids = sorted(
        thread_latest_ts, key=lambda tid: thread_latest_ts[tid], reverse=True
    )
    placeholders = ",".join("?" for _ in sorted_thread_ids)
    cur = conn.execute(
        f"SELECT thread_id, title FROM thread_titles WHERE thread_id IN ({placeholders})",
        sorted_thread_ids,
    )
    titles = dict(cur.fetchall())
    return {
        thread_id: titles.get(thread_id, "New Chat")
        for thread_id in sorted_thread_ids
    }


def save_thread_title(thread_id: str, title: str) -> None:
    conn.execute(
        "INSERT INTO thread_titles (thread_id, title) VALUES (?, ?) "
        "ON CONFLICT(thread_id) DO UPDATE SET title=excluded.title",
        (thread_id, title),
    )
    conn.commit()


def delete_thread(thread_id: str) -> None:
    for table in ["checkpoints", "writes", "checkpoint_blobs", "checkpoint_writes"]:
        try:
            conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
        except sqlite3.OperationalError:
            pass
    conn.execute("DELETE FROM thread_titles WHERE thread_id = ?", (thread_id,))
    conn.commit()
    faiss_dir = _faiss_path(thread_id)
    if os.path.exists(faiss_dir):
        shutil.rmtree(faiss_dir, ignore_errors=True)


def is_thread_empty(thread_id: str) -> bool:
    try:
        state = chatbot.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        return len(state.values.get("messages", [])) == 0
    except Exception:
        return True
