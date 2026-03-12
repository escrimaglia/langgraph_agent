# Simple agent that uses the Internal Memory (Memory Saver checkpoint) pattern.
# Saves all messages to an in-memory structure (JSON) and retrieves them on each invocation.
# By Ed SCrimaglia

from typing import List, Dict, Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, RemoveMessage
from langgraph.graph import START, END, StateGraph
from langchain_openai import AzureChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import json
from pathlib import Path

load_dotenv()

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

def multiplay(a: int, b: int) -> int:
    """
    Multiply two integers.

    Args:
        a (int): The first integer to multiply.
        b (int): The second integer to multiply.

    Returns:
        int: The product of the two integers.
    """
    return a * b

def sum(a: int, b: int) -> int:
    """
    Sum two integers.

    Args:
        a (int): The first integer to sum.
        b (int): The second integer to sum.

    Returns:
        int: The sum of the two integers.
    """
    return a + b

def devide(a: int, b: int) -> float:
    """
    Divide two integers.

    Args:
        a (int): The numerator.
        b (int): The denominator.

    Returns:
        float: The result of the division.
    """
    if b == 0:
        raise ValueError("Denominator cannot be zero.")
    return a / b

def save_graph(path: str, graph: bytes):
    file_path = Path(path).exists()
    if not file_path:
        png_graph = graph.get_graph().draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png_graph)
    else:
        print(f"Graph already exists at {path}, skipping save.")

def serialize_messages(messages: List[AnyMessage]) -> List[Dict]:
    serialized = []
    for msg in messages:
        serialized.append({
            "type": msg.__class__.__name__,
            "content": msg.content,
            "id": getattr(msg, "id", None),
            "name": getattr(msg, "name", None),
            "metadata": getattr(msg, "response_metadata", None)
        })
    return serialized

def create_filter_messages(min_messages: int = 8, keep_count: int = 6):
    def filter_messages(state: State) -> State:    
        if len(state["messages"]) <= min_messages:
            return state
        
        messages = state["messages"]        
        cutoff_index = len(messages) - keep_count
        
        while cutoff_index > 0 and cutoff_index < len(messages):
            msg = messages[cutoff_index]
            if msg.__class__.__name__ == "ToolMessage":
                cutoff_index -= 1
            else:
                break
        
        if cutoff_index > 0:
            filtered = [RemoveMessage(id=msg.id) for msg in messages[:cutoff_index]]
            print(f"Filtered out {len(filtered)} messages, keeping {len(messages) - cutoff_index} messages.")
            return {"messages": filtered}
        
        return state
    return filter_messages

# LLM Initialization
llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    temperature=0.2,
    max_completion_tokens=2000,
    max_tokens=5000,
    api_version="2024-10-21"
)

# Create Memory Saver
memory = MemorySaver()

# Bind the LLM to tools
tools = [multiplay, sum, devide]
llm_with_tools = llm.bind_tools(tools)

# System Message
sys_msg = SystemMessage(content=f"You are an helpful math assistant")

# Node to route messages based on detected tools
def calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke([sys_msg] +state["messages"])]}

filter_messages_node = create_filter_messages(min_messages=8, keep_count=6)

# Build the Graph
builder = StateGraph(State)
builder.add_node("filter_messages", filter_messages_node)
builder.add_node("calling_llm", calling_llm)
builder.add_node("tools", ToolNode(tools))

# Define edges based on tool conditions
builder.add_edge(START, "filter_messages")
builder.add_edge("filter_messages", "calling_llm") 
builder.add_conditional_edges(
    "calling_llm",
    tools_condition,
)
builder.add_edge("tools", "calling_llm")
graph = builder.compile(checkpointer=memory)

# Save the Graph
save_graph("memory_graph.png", graph)

# Define a thread to periodically save memory
session_id = "default"
config = {"configurable": {"thread_id": session_id}}

# Use the graph with messages
state = graph.invoke({"messages": [HumanMessage(content="multiplay 6 and 7", name="User")]}, config)
state = graph.invoke({"messages": [HumanMessage(content="add 10 to that", name="User")]}, config)
state = graph.invoke({"messages": [HumanMessage(content="Gracias Kiro", name="User")]}, config)

for i, msg in enumerate(state["messages"]):
        print(f"{i}: {type(msg).__name__}: {msg.content}...")

for msg in state["messages"]:
    msg.pretty_print()

# Obtener los valores del thread
channel_values = memory.get(config=config).get("channel_values")

# Serializar y guardar en un archivo JSON
serializable_data = serialize_messages(channel_values.get("messages", []))

with open(f"memory_{config.get('configurable').get('thread_id')}.json", "w", encoding="utf-8") as f:
    json.dump(serializable_data, f, indent=2, ensure_ascii=False)
