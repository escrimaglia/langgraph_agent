# Example of a simple graph with no parallel nodes. 
# Each node will execute sequentially, and the state will be passed from one node to the next.
# By Ed SCrimaglia

from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from pathlib import Path
load_dotenv()

# Define the state structure for the graph
class State(TypedDict):
    state: str

# Define a simple node that returns a value to be combined by the reducer
class ReturnNodeValue():
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: State) -> State:
        print (f"Adding {self._value} to {state['state']}")
        return {"state": self._value}

# Save the graph as a PNG file
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
builder.add_node("Node5", ReturnNodeValue("I'm Node5"))

# Flow
builder.add_edge(START, "Node1")
builder.add_edge("Node1", "Node2")
builder.add_edge("Node2", "Node3")
builder.add_edge("Node3", "Node4")
builder.add_edge("Node4", "Node5")
builder.add_edge("Node5", END)

# Compile the graph
graph = builder.compile()

# Save the Graph
save_graph("no_parallel.png", graph)

# Execute the graph
state = graph.invoke({"state": []})
print (state)

