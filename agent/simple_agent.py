# Simple agent that call LLM with tools
# By Ed SCrimaglia

from typing import List, Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langgraph.graph import START, END, StateGraph
from langchain_openai import AzureChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
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

# LLM Initialization
llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    temperature=0.2,
    max_completion_tokens=2000,
    max_tokens=5000,
    api_version="2024-10-21"
)

# Bind the LLM to tools
tools = [multiplay, sum, devide]
llm_with_tools = llm.bind_tools(tools)

# System Message
sys_msg = SystemMessage(content=f"You are an helpful math assitant")

# Node to route messages based on detected tools
def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build the Graph
builder = StateGraph(State)
builder.add_node("calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

# Define edges based on tool conditions
builder.add_edge(START, "calling_llm")
builder.add_conditional_edges(
    "calling_llm",
    tools_condition,
)
builder.add_edge("tools", "calling_llm")
builder.add_edge("calling_llm", END)
graph = builder.compile()

# Save the Graph
save_graph("agent_graph.png", graph)

# Use the graph with initial messages
state = graph.invoke({"messages": [HumanMessage(content="Multiply 6 and 7. Add 10 to the result and then divide that by 2.")]})
for message in state["messages"]:
    message.pretty_print()
