# This code demonstrates how to implement a memory management system for a conversational agent using LangGraph. 
# The agent can perform mathematical operations using tools and manages its memory by filtering out old messages when the number of messages exceeds a certain threshold.
# By fltering messages, the agent can maintain a more relevant context for the conversation while keeping the memory usage in check saving tokens and improving performance.
# The graph is built with nodes for invoking the LLM, routing based on tool conditions, and filtering messages, and it uses a SqliteSaver to persist the memory state.
# MessageState class is used instead of a TypedDict class as used in other examples.
# By Ed Scrimaglia

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, RemoveMessage
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from pathlib import Path

load_dotenv()

# Tools without parameters to access the state
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
    file_exist = Path(path).exists()
    if not file_exist:
        png_graph = graph.get_graph().draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png_graph)
    else:
        print(f"Graph already exists at {path}, skipping save.")

# Define a function to filter messages when the number of messages exceeds a certain threshold
def create_filter_messages(min_messages: int = 8, keep_count: int = 6):
    def filter_messages(state: MessagesState) -> MessagesState:
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
def calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# route after LLM to decide the next node based on tool conditions
def route_after_llm(state: MessagesState) -> str:
    next_node = tools_condition(state)
    if next_node == END:
        return "filtering_messages"
    return next_node

# Create the filter messages node with specific parameters
filter_messages_node = create_filter_messages(min_messages=12, keep_count=10)

# Build the Graph
builder = StateGraph(MessagesState)
builder.add_node("calling_llm", calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_node("filtering_messages", filter_messages_node)

# Define edges based on tool conditions
builder.add_edge(START, "calling_llm")
builder.add_conditional_edges(
    "calling_llm",
    route_after_llm,
    {
        "tools": "tools",
        "filtering_messages": "filtering_messages",
    },
)
builder.add_edge("tools", "calling_llm")
builder.add_edge("filtering_messages", END)

# Compile the graph with the memory checkpointer
graph = builder.compile(checkpointer=memory)

# Save the Graph
save_graph("memory_summary.png", graph)

# Define a thread to periodically save memory
config = {"configurable": {"thread_id": session_id}}

# Get the initial state of the graph to check the number of messages stored in memory
graph_state = graph.get_state(config=config)
print(f"Initial State Messages: {len(graph_state.values.get('messages', []))}")

# Use the graph with messages
state = []
while True:
    query = input("->Prompt (or 'exit' to quit): ")
    if query.lower() in ["exit", "quit"]:
        break
    state = graph.invoke({"messages": [HumanMessage(content=query, name="User")]}, config)
    print (f"--> Respuesta: {state['messages'][-1].content}")

if len(state) > 0:
    print (f"\nFinal State Messages:")
    for i, msg in enumerate(state["messages"]):
        print(f"{i}: {type(msg).__name__}: {msg.content}...")

# for msg in state["messages"]:
#     msg.pretty_print()
