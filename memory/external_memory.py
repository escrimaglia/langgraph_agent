# Simple agent that uses the External Memory (Sqlite) pattern.
# Saves all messages to a Sqlite database and retrieves them on each invocation.
# By Ed SCrimaglia

from typing import List, Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langgraph.graph import START, END, StateGraph
from langchain_openai import AzureChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from pathlib import Path 

load_dotenv()

# Define the state structure for the graph
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

# Define some simple tools for the agent to use
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

# Save the graph as a PNG file 
def save_graph(path: str, graph: bytes):
    file_path = Path(path).exists()
    if not file_path:
        png_graph = graph.get_graph().draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png_graph)
    else:
        print(f"Graph already exists at {path}, skipping save.")


# LLM Initialization
llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    temperature=0.2,
    max_completion_tokens=2000,
    max_tokens=5000,
    api_version="2024-10-21"
)

# Define a unique session ID
session_id = "default"

# Path for the Sqlite database
db_path = f"{session_id}_memory.db"
conn = sqlite3.connect(db_path, check_same_thread=False)

# Check to Sqlite3 Memory Saver
memory = SqliteSaver(conn)

# Bind the LLM to tools
tools = [multiplay, sum, devide]
llm_with_tools = llm.bind_tools(tools)

# System Message
sys_msg = SystemMessage(content=f"You are an helpful math assistant")

# Node to route messages based on detected tools
def calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke([sys_msg] +state["messages"])]}

# Build the Graph
builder = StateGraph(State)
builder.add_node("calling_llm", calling_llm)
builder.add_node("tools", ToolNode(tools))

# Define edges based on tool conditions
builder.add_edge(START, "calling_llm") 
builder.add_conditional_edges(
    "calling_llm",
    tools_condition,
)
builder.add_edge("tools", "calling_llm")
graph = builder.compile(checkpointer=memory)

# Save the Graph
save_graph("memory_graph.png", graph)

# Define a thread to periodically save memory
config = {"configurable": {"thread_id": session_id}}

graph_state = graph.get_state(config=config)
# print (graph_state.values.get("messages", []))
print(f"Initial State Messages: {len(graph_state.values.get('messages', []))}")

# Use the graph with messages
state = graph.invoke({"messages": [HumanMessage(content="multiplay 6 and 7", name="User")]}, config)
state = graph.invoke({"messages": [HumanMessage(content="add 10 to that", name="User")]}, config)
state = graph.invoke({"messages": [HumanMessage(content="Gracias Kiro", name="User")]}, config)
state = graph.invoke({"messages": [HumanMessage(content="Hola Kiro, soy Ed", name="User")]}, config)
state = graph.invoke({"messages": [HumanMessage(content="Kiro, sabes quien soy?", name="User")]}, config)

for i, msg in enumerate(state["messages"]):
        print(f"{i}: {type(msg).__name__}: {msg.content}...")

for msg in state["messages"]:
    msg.pretty_print()
