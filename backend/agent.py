from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from rag import search_docs
import os, datetime, time
from dotenv import load_dotenv

load_dotenv()

# --- Tools ---
@tool
def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str) -> str:
    """Evaluates a math expression. Example: 2**10 or sqrt(144)"""
    import math
    try:
        result = eval(expression, {"__builtins__": {}}, vars(math))
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def search_knowledge(query: str) -> str:
    """Search the knowledge base for relevant information."""
    result = search_docs(query)
    return result if result else "No relevant information found in knowledge base."

TOOLS = [get_current_time, calculate, search_knowledge]

# Optional: live web search via Tavily (latest news / current events).
# Only added to the toolset if a TAVILY_API_KEY is present, so the app
# still runs fine without it.
tavily_key = os.getenv("TAVILY_API_KEY")
if tavily_key:
    try:
        from langchain_tavily import TavilySearch
        tavily_tool = TavilySearch(max_results=5, tavily_api_key=tavily_key)
        tavily_tool.name = "web_search"
        tavily_tool.description = (
            "Search the live web for current events, news, prices, or anything "
            "that requires up-to-date information beyond the model's training data."
        )
        TOOLS.append(tavily_tool)
    except ImportError:
        print("⚠️  langchain-tavily not installed — run: pip install langchain-tavily")

# --- LLM setup ---
# Provider priority: Groq first (free tier, LPU inference — by far the fastest
# option for open models like gpt-oss-120b), then OpenRouter free models as
# a fallback chain if Groq is unavailable/unconfigured/rate-limited.
groq_key = os.getenv("GROQ_API_KEY")
openrouter_key = os.getenv("OPENROUTER_API_KEY")

if not groq_key and not openrouter_key:
    raise ValueError("Set GROQ_API_KEY and/or OPENROUTER_API_KEY in your .env file!")

# Each entry: (base_url, api_key, model_name)
PROVIDERS = []

if groq_key:
    PROVIDERS += [
        ("https://api.groq.com/openai/v1", groq_key, "openai/gpt-oss-120b"),
        ("https://api.groq.com/openai/v1", groq_key, "openai/gpt-oss-20b"),
        ("https://api.groq.com/openai/v1", groq_key, "llama-3.3-70b-versatile"),
    ]

if openrouter_key:
    PROVIDERS += [
        ("https://openrouter.ai/api/v1", openrouter_key, "meta-llama/llama-3.1-8b-instruct:free"),
        ("https://openrouter.ai/api/v1", openrouter_key, "qwen/qwen-2.5-7b-instruct:free"),
        ("https://openrouter.ai/api/v1", openrouter_key, "google/gemma-3-12b-it:free"),
        ("https://openrouter.ai/api/v1", openrouter_key, "openrouter/free"),
    ]

# Kept for backwards compatibility with anything importing FALLBACK_MODELS.
FALLBACK_MODELS = [p[2] for p in PROVIDERS]

def get_llm_with_tools(base_url: str, api_key: str, model_name: str):
    llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        streaming=True,
        request_timeout=30,
    )
    return llm.bind_tools(TOOLS)

# --- State ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

SYSTEM_PROMPT = """You are Voxa AI, a helpful AI assistant built by Chandan Singh.
You have access to tools: get time, calculate math, and search a knowledge base.
Always use the search_knowledge tool when users ask about what you can do or who made you.
Be concise and clear in your responses."""

def chatbot_node(state: AgentState):
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    last_error = None

    for base_url, api_key, model_name in PROVIDERS:
        # Give each model up to 2 attempts (helps with brief rate-limit blips)
        for attempt in range(2):
            try:
                llm_with_tools = get_llm_with_tools(base_url, api_key, model_name)
                response = llm_with_tools.invoke(messages)
                return {"messages": [response]}
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "404" in err_str or "unavailable" in err_str:
                    if attempt == 0:
                        time.sleep(1.5)  # short wait before retrying same model once
                    continue  # move to next attempt or next model
                raise  # unrelated error (e.g. bad API key) — stop immediately

    # All providers/models failed
    raise last_error if last_error else Exception("All free models are currently busy. Please try again shortly.")

# --- Build Graph ---
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("chatbot", chatbot_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.set_entry_point("chatbot")
    graph.add_conditional_edges("chatbot", tools_condition)
    graph.add_edge("tools", "chatbot")
    return graph.compile()

agent = build_graph()


# ── Streaming helper ──────────────────────────────────────────────────
# Two-phase approach that works on every device and provider:
# Phase 1 — invoke() to fully resolve any tool calls (time, calculator, RAG)
#           so raw tool-call JSON never leaks into the streamed output.
# Phase 2 — llm.stream() to yield the final answer token-by-token for a
#           real typewriter effect on all browsers including Safari/iOS.
def stream_chat(history_messages):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history_messages
    last_error = None

    for base_url, api_key, model_name in PROVIDERS:
        for attempt in range(2):
            try:
                llm_with_tools = get_llm_with_tools(base_url, api_key, model_name)

                # Phase 1: resolve tool calls (non-streaming, fast)
                result = agent.invoke({"messages": messages})
                final_messages = result["messages"]

                # Phase 2: stream the final answer token by token
                # Re-invoke with the resolved messages so the LLM streams
                # its answer without needing to call tools again.
                llm_plain = ChatOpenAI(
                    base_url=base_url,
                    api_key=api_key,
                    model=model_name,
                    streaming=True,
                    request_timeout=30,
                )
                got_any = False
                for chunk in llm_plain.stream(final_messages):
                    if chunk.content:
                        got_any = True
                        yield chunk.content
                if got_any:
                    return

                # Fallback: if streaming yielded nothing, return the
                # already-resolved reply as one chunk so nothing is lost.
                reply = final_messages[-1].content
                if reply:
                    yield reply
                return

            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "404" in err_str or "unavailable" in err_str:
                    if attempt == 0:
                        time.sleep(1.5)
                    continue
                raise

    raise last_error if last_error else Exception("All free models are currently busy. Please try again shortly.")


# ── RAG-only mode ─────────────────────────────────────────────────────
# Used by the /chat/rag route. Answers ONLY from your docs/ knowledge base,
# no tool calls, no general LLM knowledge. Plain (non-streaming) call,
# matching the style of chatbot_node above.

RAG_SYSTEM_PROMPT = """You are Voxa AI in RAG-only mode.
You must answer ONLY using the provided knowledge base context below.
If the context does not contain the answer, respond exactly with:
"I don't have information about that in my knowledge base."
Do not use any outside knowledge. Do not guess. Do not use other tools.

Knowledge base context:
{context}
"""

def get_rag_response(user_query: str, history_messages=None) -> str:
    """
    Returns a plain string answer generated strictly from the RAG knowledge base.
    Uses the same FALLBACK_MODELS list and retry pattern as chatbot_node.
    """
    context = search_docs(user_query, k=4)

    if not context:
        return "I don't have information about that in my knowledge base."

    rag_prompt = RAG_SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=rag_prompt)]
    if history_messages:
        messages += history_messages
    messages.append(SystemMessage(content=f"User question: {user_query}"))

    last_error = None

    for base_url, api_key, model_name in PROVIDERS:
        for attempt in range(2):
            try:
                llm = ChatOpenAI(
                    base_url=base_url,
                    api_key=api_key,
                    model=model_name,
                    streaming=False,
                    request_timeout=30,
                )
                response = llm.invoke(messages)
                return response.content
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "404" in err_str or "unavailable" in err_str:
                    if attempt == 0:
                        time.sleep(1.5)
                    continue
                raise

    raise last_error if last_error else Exception("All free models are currently busy. Please try again shortly.")


def stream_rag_response(user_query: str, history_messages=None):
    """Token-level streaming version of get_rag_response."""
    context = search_docs(user_query, k=4)
    if not context:
        yield "I don't have information about that in my knowledge base."
        return

    rag_prompt = RAG_SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=rag_prompt)]
    if history_messages:
        messages += history_messages
    messages.append(SystemMessage(content=f"User question: {user_query}"))

    last_error = None
    for base_url, api_key, model_name in PROVIDERS:
        for attempt in range(2):
            try:
                llm = ChatOpenAI(
                    base_url=base_url,
                    api_key=api_key,
                    model=model_name,
                    streaming=True,
                    request_timeout=30,
                )
                got_any = False
                for chunk in llm.stream(messages):
                    if chunk.content:
                        got_any = True
                        yield chunk.content
                if got_any:
                    return
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "404" in err_str or "unavailable" in err_str:
                    if attempt == 0:
                        time.sleep(1.5)
                    continue
                raise

    raise last_error if last_error else Exception("All free models are currently busy. Please try again shortly.")
