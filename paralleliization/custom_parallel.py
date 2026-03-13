# Example of a custom parallel graph with a custom reducer function.
# Parallel nodes will return a list of values, and the custom reducer will combine and sort these lists.
# Change the custom_reducer function by 'operator.add' to see how it affects the final output.
# By Ed Scrimaglia

from typing import List, Dict, Annotated
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import START, END, StateGraph
from langgraph.errors import InvalidUpdateError
from pathlib import Path
import operator
load_dotenv()

# Define a custom reducer function to combine and sort the results from parallel nodes
def custom_reducer(left, right):
    """ combina y ordena los elementos de las listas left y right """
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return sorted(left + right, reverse=False)

# Define the state structure for the graph
class State(TypedDict):
    state: Annotated[List, custom_reducer]
    # state: Annotated[List, operator.add]

# Define a simple node that returns a value to be combined by the reducer
class ReturnNodeValue():
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: State) -> State:
        print (f"Adding {self._value} to {state['state']}")
        return {"state": [self._value]}

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
builder.add_node("Node2b", ReturnNodeValue("I'm Node2b"))
builder.add_node("Node3", ReturnNodeValue("I'm Node3"))
builder.add_node("Node4", ReturnNodeValue("I'm Node4"))

# Flow
builder.add_edge(START, "Node1")
builder.add_edge("Node1", "Node2")
builder.add_edge("Node1", "Node3")
builder.add_edge("Node2", "Node2b")
builder.add_edge(["Node2b", "Node3"], "Node4")
builder.add_edge("Node4", END)

# Compile the graph
graph = builder.compile()

# Save the Graph
save_graph("custom_parallel.png", graph)

# Execute the graph
try:
    state = graph.invoke({"state": []})
except InvalidUpdateError as error:
    print(f"Error invoking graph: {error}")
print (state["state"])



