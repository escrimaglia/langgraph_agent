# Example of a simple router pattern graph that routes messages to different nodes based on detected tools.
# add_messages reducer is used to combine messages from different nodes, but you can create your own custom reducer for more complex routing logic.
# By Ed Scrimaglia

from typing import List, Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AnyMessage
from langgraph.graph import START, END, StateGraph
from langchain_openai import AzureChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pathlib import Path

load_dotenv()

# Define the state with a custom reducer to combine messages
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

# Define a simple tool function
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

# Save the graph as a PNG file
def save_graph(path: str, graph: bytes):
    file_exist = Path(path).exists()
    if not file_exist:
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
llm_with_tools = llm.bind_tools([multiplay])

# Node to route messages based on detected tools
def calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build the Graph
builder = StateGraph(State)
builder.add_node("calling_llm", calling_llm)
builder.add_node("tools", ToolNode(tools=[multiplay]))

# Define edges based on tool conditions
builder.add_edge(START, "calling_llm")
builder.add_conditional_edges(
    "calling_llm",
    tools_condition,
)
builder.add_edge("tools", END)
graph = builder.compile()

save_graph("router.png", graph)

# Will route to the tool node if the tool condition is met, otherwise it will just return the LLM response.
state = graph.invoke({"messages": [HumanMessage(content="Multiply 6 and 7")]})
for message in state["messages"]:
    message.pretty_print()

state = graph.invoke({"messages": [HumanMessage(content="What is the capital of France?")]})
for message in state["messages"]:
    message.pretty_print()
