# Example of a simple parallel graph.
# Nodes "Node2" and "Node3" will execute in parallel, and their outputs will be combined by the reducer function.
# By Ed Scrimaglia

from typing import List, Annotated
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.errors import InvalidUpdateError
from pathlib import Path
import operator
load_dotenv()

class State(TypedDict):
    state: Annotated[List, operator.add]

class ReturnNodeValue():
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: State) -> State:
        print (f"Adding {self._value} to {state['state']}")
        return {"state": [self._value]}

# Save the graph
def save_graph(path: str, graph: bytes):
    file_exist = Path(path).exists()
    if not file_exist:
        png_graph = graph.get_graph().draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png_graph)
    else:
        print(f"Graph already exists at {path}, skipping save.")

# Build the Graph
builder = StateGraph(State)

# Add the nodes
builder.add_node("Node1", ReturnNodeValue("I'm Node1"))
builder.add_node("Node2", ReturnNodeValue("I'm Node2"))
builder.add_node("Node3", ReturnNodeValue("I'm Node3"))
builder.add_node("Node4", ReturnNodeValue("I'm Node4"))

# Flow
builder.add_edge(START, "Node1")
builder.add_edge("Node1", "Node2")
builder.add_edge("Node1", "Node3")
builder.add_edge(["Node2", "Node3"], "Node4")
builder.add_edge("Node4", END)

# Compile the graph
graph = builder.compile()

# Save the Graph
save_graph("parallel.png", graph)

# Execute the graph
try:
    state = graph.invoke({"state": []})
except InvalidUpdateError as error:
    print(f"Error invoking graph: {error}")
print (state["state"])



