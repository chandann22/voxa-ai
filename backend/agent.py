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
 
# --- LLM setup ---
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file!")
 
BASE_URL = "https://openrouter.ai/api/v1"
 
# Ordered list of free models to try — least crowded / most reliable first.
# If one is rate-limited or unavailable, the next one is tried automatically.
FALLBACK_MODELS = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "qwen/qwen-2.5-7b-instruct:free",
    "google/gemma-3-12b-it:free",
    "openrouter/free",
]
 
def get_llm_with_tools(model_name: str):
    llm = ChatOpenAI(
        base_url=BASE_URL,
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
 
    for model_name in FALLBACK_MODELS:
        # Give each model up to 2 attempts (helps with brief rate-limit blips)
        for attempt in range(2):
            try:
                llm_with_tools = get_llm_with_tools(model_name)
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
 
    # All models in the fallback list failed
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