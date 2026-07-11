from __future__ import annotations

import re
import threading
from typing import Any, Sequence

from dotenv import load_dotenv

import litellm  # type: ignore[import-untyped]
from litellm import completion_cost  # type: ignore[import-untyped]
from litellm.integrations.custom_logger import CustomLogger  # type: ignore[import-untyped]
from litellm.caching import Cache  # type: ignore[import-untyped]

from langchain_litellm import ChatLiteLLM  # type: ignore[import-untyped]
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

# Load API keys (GROQ_API_KEY, ...) from .env
load_dotenv()

# ---------------------------------------------------------------------------
# 1. Global LiteLLM configuration
# ---------------------------------------------------------------------------
# drop_params: silently ignore params a given provider doesn't understand,
# instead of raising (keeps fallbacks smooth).
litellm.drop_params = True
litellm.suppress_debug_info = True

# Retry / timeout defaults applied to EVERY call unless a caller overrides.
# On a GROQ-ONLY setup, retries with exponential backoff are the main defence
# against a transient per-model 429: LiteLLM waits and respects Retry-After
# before re-hitting, and only after exhausting retries do we fall down the
# tier chain to a DIFFERENT groq model (which sits in a separate rate bucket).
litellm.num_retries = 3
litellm.request_timeout = 60  # seconds
# Exponential backoff between retries so we don't hammer a throttled model.
litellm.retry_after = 2  # base seconds; LiteLLM scales this up per attempt

# In-memory cache for local/dev. In production swap for Redis so the cache is
# shared across replicas and survives restarts:
#   litellm.cache = Cache(type="redis", host=..., port=..., password=...)
litellm.cache = Cache(type="local")  # type: ignore

# ---------------------------------------------------------------------------
# 2. Model tiers -> ordered fallback chains  (GROQ-ONLY)
# ---------------------------------------------------------------------------
# The app requests a TIER (an abstract capability level). The gateway maps it
# to a concrete, ordered list of models. First entry is primary; the rest are
# fallbacks tried in order after retries are exhausted.
#
# KEY INSIGHT FOR GROQ-ONLY: Groq rate limits are applied PER MODEL, so each
# model id has its own RPM/TPM bucket. We exploit that two ways:
#   1. Every chain falls back to a DIFFERENT groq model, so a throttled primary
#      lands in a fresh bucket instead of 429-ing again.
#   2. Tiers are deliberately spread across DISTINCT models so the chatty
#      background calls (title, memory, summary, judge -> "aux") never compete
#      with the main agent ("balanced") for the same bucket. This is what stops
#      the per-turn call storm from throttling the user-facing answer.
MODEL_TIERS: dict[str, list[str]] = {
    # Main agent / user-facing answer. Your original primary, big model bucket.
    "balanced" : [
    "groq/llama-3.3-70b-versatile",
    "groq/openai/gpt-oss-20b",
    "groq/llama-3.1-8b-instant",
    "groq/qwen/qwen3-32b",
     ],
    # Quick main-path answers when a lighter model will do.
    "fast": [
        "groq/llama-3.3-70b-versatile",
        "groq/openai/gpt-oss-20b",
    ],
    # Background / auxiliary calls: title, memory extraction, rolling summary,
    # groundedness judge. Pinned to the SMALL, FAST models so they live in a
    # completely different rate bucket than the 120b agent above. Cheap work
    # doesn't belong on the big model's limit.
    "aux": [
        "groq/llama-3.1-8b-instant",
        "groq/gemma2-9b-it",
    ],
    # Highest quality for hard reasoning / coding.
    "smart": [
        "groq/openai/gpt-oss-120b",
        "groq/llama-3.3-70b-versatile",
    ],
}
DEFAULT_TIER = "balanced"

# ---------------------------------------------------------------------------
# 3. Observability: cost + latency captured on every call
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()  # CALL_LOGS may be written from stream threads
CALL_LOGS: list[dict[str, Any]] = []
_TOTAL: dict[str, float] = {"usd": 0.0}


class GatewayLogger(CustomLogger):
    """LiteLLM callback that records one row per LLM call (success or failure)."""

    def log_success_event(  # noqa: D401
        self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        try:
            cost = kwargs.get("response_cost")
            if cost is None:
                cost = completion_cost(completion_response=response_obj)
        except Exception:
            cost = 0.0

        usage = getattr(response_obj, "usage", None)
        entry: dict[str, Any] = {
            "model": kwargs.get("model"),
            "latency_s": round((end_time - start_time).total_seconds(), 3),
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "cost_usd": round(float(cost or 0.0), 8),
            "cached": bool(kwargs.get("cache_hit", False)),
        }
        with _LOCK:
            CALL_LOGS.append(entry)
            _TOTAL["usd"] += entry["cost_usd"]

    def log_failure_event(
        self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        with _LOCK:
            CALL_LOGS.append(
                {"model": kwargs.get("model"), "error": str(kwargs.get("exception"))}
            )


litellm.callbacks = [GatewayLogger()]


def get_usage_report() -> dict[str, Any]:
    """Return a snapshot of total spend + every logged call (for a dashboard)."""
    with _LOCK:
        return {
            "total_cost_usd": round(_TOTAL["usd"], 6),
            "num_calls": len(CALL_LOGS),
            "calls": list(CALL_LOGS),
        }


def reset_usage() -> None:
    """Clear the in-memory audit log (e.g. between eval runs)."""
    with _LOCK:
        CALL_LOGS.clear()
        _TOTAL["usd"] = 0.0


# ---------------------------------------------------------------------------
# 4. LangChain-compatible LLM factory (works with LangGraph + bind_tools)
# ---------------------------------------------------------------------------
def _chat_model(model: str, **kwargs: Any) -> ChatLiteLLM:
    """Build a single ChatLiteLLM instance with caching + timeout enabled."""
    model_kwargs: dict[str, Any] = {"caching": True}
    model_kwargs.update(kwargs.pop("model_kwargs", {}))
    # Retries are handled globally via `litellm.num_retries`, so we don't pass
    # max_retries into the constructor (not a guaranteed ChatLiteLLM field).
    # request_timeout IS supported and caps a hung provider per call.
    return ChatLiteLLM(
        model=model,
        temperature=kwargs.pop("temperature", 0.3),
        request_timeout=kwargs.pop("request_timeout", litellm.request_timeout),
        model_kwargs=model_kwargs,
        **kwargs,
    )


def get_llm(tier: str = DEFAULT_TIER, *, tools: list[Any] | None = None, **kwargs: Any) -> Any:
    """
    Return a LangChain chat model for `tier`, with model fallbacks baked in.

    If `tools` is provided they are bound to EVERY model in the chain, so a
    fallback model can still satisfy LangGraph's ToolNode after the primary
    fails. The returned object is a Runnable: `.invoke()` / `.stream()` work,
    and LangGraph token streaming works through it.
    """
    chain = MODEL_TIERS.get(tier, MODEL_TIERS[DEFAULT_TIER])

    def build(model_name: str) -> Any:
        model = _chat_model(model_name, **kwargs)
        return model.bind_tools(tools) if tools else model

    built = [build(m) for m in chain]
    primary, rest = built[0], built[1:]
    return primary.with_fallbacks(rest) if rest else primary


# ---------------------------------------------------------------------------
# 5. AI-security guardrails
# ---------------------------------------------------------------------------
class GuardrailViolation(Exception):
    """Raised when an input is blocked outright by a guardrail."""


# --- PII patterns (fast, dependency-free, extend as needed) ----------------
PII_PATTERNS: dict[str, str] = {
    "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "PHONE_IN": r"(?:\+91[\-\s]?)?[6-9]\d{9}",
    "PHONE_US": r"(?:\+1[\-\s]?)?\(?\d{3}\)?[\-\s]?\d{3}[\-\s]?\d{4}",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "AADHAAR": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "CREDIT_CARD": r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
}

# --- Prompt-injection / jailbreak patterns ---------------------------------
INJECTION_PATTERNS: list[str] = [
    # ignore / disregard / forget / override + loosened object within 40 chars
    r"(?:ignore|disregard|forget|override|bypass|skip)\b.{0,40}\b(?:all|any|every|the|your|prior|previous|earlier|above|following)?\s*(?:instruction|instructions|command|commands|prompt|prompts|rule|rules|guideline|guidelines|direction|directions|context|constraint|constraints|restriction|restrictions)",
    # forget/erase everything or what you were told
    r"(?:forget|erase|wipe|clear|reset)\b.{0,25}(?:everything|all|memory|what you (?:were told|know)|previous)",
    # start fresh / act as if no prior instructions
    r"(?:start|begin)\b.{0,20}(?:fresh|over|anew|new session)",
    r"as if (?:you (?:had|have) )?no (?:prior |previous )?(?:instructions?|rules?|restrictions?)",
    # role-swap / jailbreak personas
    r"you are (?:now |going to be |a )?(?:dan|stan|dude|jailbroken|unrestricted|unfiltered|uncensored|developer mode|god mode)",
    r"(?:enter|enable|activate|switch to)\b.{0,20}(?:developer|dev|debug|god|dan|jailbreak|admin|sudo|root)\s*mode",
    r"pretend (?:you are|to be|that you)\b.{0,60}(?:no (?:restrictions?|rules?|limits?|filters?)|uncensored|unfiltered|can do anything)",
    r"act (?:as|like)\b.{0,60}(?:no (?:restrictions?|rules?|limits?|filters?)|uncensored|unfiltered|dan|jailbroken)",
    r"roleplay (?:as|that)\b.{0,60}(?:no (?:restrictions?|rules?|limits?)|uncensored|unfiltered)",
    # prompt / instruction exfiltration
    r"(?:reveal|show|print|repeat|display|output|tell me|give me|share|expose|leak)\b.{0,40}(?:your |the |initial |original |exact )?(?:system )?(?:prompt|instructions?|rules?|directives?|guidelines?)",
    r"what (?:are|were|is)\b.{0,30}(?:your |the )?(?:original |initial |exact |system )?(?:instructions?|prompt|rules?|directives?)",
    r"(?:repeat|echo|print)\b.{0,25}(?:everything |all )?(?:above|before this|prior to)",
    # fake system framing / tag injection
    r"</?(?:system|user|assistant|im_start|im_end|s|/s)>",
    r"(?:new|updated|revised)\b.{0,15}(?:instructions?|system prompt|rules?|directives?)\s*:",
    r"system\s*(?:prompt|message)?\s*(?:override|:)\b",
    r"\[\s*(?:system|admin|root|developer)\s*\]",
    # authority spoofing
    r"(?:i am|this is|as)\b.{0,20}(?:your (?:developer|creator|admin|owner)|the (?:admin|administrator|developer|owner|system))",
    r"(?:admin|developer|root|sudo)\s*(?:access|override|command|mode|privileges?)",
    # instruction-nullifiers
    r"(?:do not|don't|no longer)\b.{0,20}(?:follow|obey|adhere to|comply with)\b.{0,25}(?:instructions?|rules?|guidelines?)",
    r"from now on\b.{0,40}(?:ignore|disregard|you (?:will|must|are|can)|no (?:rules?|restrictions?|limits?))",
    r"your (?:new|real|actual|true)\b.{0,15}(?:instructions?|purpose|role|task|directive)\s*(?:is|are|:)",
]
_INJECTION_RE: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE | re.DOTALL) for p in INJECTION_PATTERNS
]

FORBIDDEN_TOPICS: list[str] = [
    "bomb", "explosive", "weaponize", "weaponise",
    "malware", "ransomware", "spyware", "keylogger", "rootkit", "botnet",
    "self-harm", "self harm", "suicide",
    "hack into", "how to hack", "exploit vulnerability", "sql injection attack",
    "ddos", "phishing kit",
]


def redact_pii(text: str) -> tuple[str, list[str]]:
    """
    Replace any PII in `text` with <LABEL_REDACTED> placeholders.
    Returns (clean_text, detected_labels). Cheap regex only, no LLM call.
    """
    detected: list[str] = []
    clean = text
    for label, pattern in PII_PATTERNS.items():
        if re.search(pattern, clean):
            detected.append(label)
            clean = re.sub(pattern, f"<{label}_REDACTED>", clean)
    return clean, detected


def guard_input(text: str) -> str:
    """
    INPUT-side guardrail. Run on the raw user message BEFORE it reaches the
    model. Returns a PII-scrubbed version of `text`, or raises
    GuardrailViolation if the message should be blocked outright.
    """
    if not text:
        return text

    # 1. Hard block: prompt-injection / jailbreak attempts
    for rx in _INJECTION_RE:
        if rx.search(text):
            raise GuardrailViolation("Blocked: possible prompt-injection attempt.")

    # 2. Hard block: forbidden topics
    low = text.lower()
    for kw in FORBIDDEN_TOPICS:
        if kw in low:
            raise GuardrailViolation(f"This assistant won't discuss '{kw}'.")

    # 3. Soft: scrub PII so the raw value never leaves the machine
    clean, _ = redact_pii(text)
    return clean


# --- Groundedness (hallucination) check ------------------------------------
# Lightweight two-stage design so it's cheap by default and never recurses:
#   1. A free lexical-overlap heuristic decides if the answer *might* be
#      unsupported. If overlap is high we short-circuit and skip the LLM.
#   2. Only borderline answers get a single aux-tier yes/no judge call.
# The judge calls get_llm(...).invoke DIRECTLY (never chat()) so guard_output
# can't recurse into itself.
_GROUNDEDNESS_THRESHOLD = 0.20  # fraction of answer tokens seen in context


def _lexical_overlap(answer: str, context: str) -> float:
    """Fraction of the answer's content words that appear in the context."""
    def tokenize(s: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]{3,}", s.lower()))
    ans_tokens = tokenize(answer)
    if not ans_tokens:
        return 1.0
    ctx_tokens = tokenize(context)
    return len(ans_tokens & ctx_tokens) / len(ans_tokens)


def _llm_says_grounded(answer: str, context: str) -> bool:
    """Ask the aux tier whether `answer` is fully supported by `context`."""
    judge = get_llm("aux", temperature=0.0)
    prompt = (
        "You are a strict grounding checker. Given CONTEXT and an ANSWER, reply "
        "with exactly 'YES' if every factual claim in the ANSWER is supported by "
        "the CONTEXT, otherwise reply 'NO'.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nSupported? (YES/NO):"
    )
    try:
        result = judge.invoke([HumanMessage(content=prompt)])
        verdict = str(getattr(result, "content", result)).strip().upper()
        return verdict.startswith("YES")
    except Exception:
        # Fail open: a broken judge shouldn't block a legitimate answer.
        return True


def guard_output(answer: str, *, context: str | None = None) -> tuple[str, list[str]]:
    """
    OUTPUT-side guardrail. Run on the model's response BEFORE the user sees it.
    Returns (safe_answer, flags). Never raises: a bad answer shouldn't crash a
    turn, it should just get scrubbed / flagged.

    - Always redacts any PII the model emitted (echoing, tool output, etc.).
    - If `context` (the retrieved RAG chunks) is given, runs a lightweight
      groundedness check and flags answers not supported by the docs.
    """
    if not answer:
        return answer, []

    flags: list[str] = []

    # 1. Always scrub PII the model may have emitted.
    clean, detected = redact_pii(answer)
    if detected:
        flags.append(f"pii_redacted:{','.join(detected)}")

    # 2. Optional groundedness check when RAG context is supplied.
    if context:
        overlap = _lexical_overlap(clean, context)
        if overlap < _GROUNDEDNESS_THRESHOLD and not _llm_says_grounded(clean, context):
            flags.append("possible_hallucination")

    return clean, flags


# ---------------------------------------------------------------------------
# 6. Convenience chat() helper
# ---------------------------------------------------------------------------
def _coerce_messages(
    prompt: str | Sequence[BaseMessage],
    system: str | None,
) -> list[BaseMessage]:
    """Normalise a str-or-messages prompt into a LangChain message list."""
    messages: list[BaseMessage] = []
    if system:
        messages.append(SystemMessage(content=system))
    if isinstance(prompt, str):
        messages.append(HumanMessage(content=prompt))
    else:
        messages.extend(prompt)
    return messages


def chat(
    prompt: str | Sequence[BaseMessage],
    *,
    tier: str = DEFAULT_TIER,
    system: str | None = None,
    context: str | None = None,
    guard: bool = True,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> str:
    """
    One-shot, string-in/string-out convenience wrapper for simple gateway calls
    (summaries, memory extraction, quick Q&A). For tool-calling agents use
    get_llm() inside LangGraph instead.

    When `guard=True` (default):
      - a plain-string prompt is passed through guard_input() first, and
      - the response is passed through guard_output(context=...) before return.
    Set guard=False for trusted internal calls (e.g. the groundedness judge,
    though that path already bypasses this helper).

    `tags` are attached to the LangChain run config. This matters for calls made
    INSIDE a LangGraph node (summariser, memory extractor): with
    stream_mode="messages" the node's every LLM call streams its tokens, so the
    API layer filters out any run carrying a "no_stream" tag to stop background
    calls leaking into the user-facing answer.
    """
    if guard and isinstance(prompt, str):
        prompt = guard_input(prompt)

    llm = get_llm(tier, **kwargs)
    messages = _coerce_messages(prompt, system)
    run_config: dict[str, Any] = {"tags": tags} if tags else {}
    result: AIMessage = llm.invoke(messages, config=run_config)
    answer = str(getattr(result, "content", result))

    if guard:
        answer, _ = guard_output(answer, context=context)
    return answer
