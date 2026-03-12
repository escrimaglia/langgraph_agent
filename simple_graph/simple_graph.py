# This example demonstrates how to create a simple state graph using the LangGraph library. 
# The graph consists of a few nodes that manipulate a string in the state, and a decision node that routes the flow based on a random condition. 
# The graph is saved as a PNG file, and then executed with an initial state to see the final output.
# By Ed Scrimaglia

from typing import TypedDict, Literal
import random
import os
from langgraph.graph import StateGraph, START, END
from pathlib import Path

# Class to define the state schema for the graph. In this case, we have a simple state that holds a string.
class State(TypedDict):
    graph_state: str

# Define the nodes of the graph. Each node is a function that takes the current state and returns an updated state.
def node1(state: State) -> State:
    print ("--- Node 1 executed ---")
    return {"graph_state": state["graph_state"] + "I am"}

def node2(state: State) -> State:
    print ("--- Node 2 executed ---")
    return {"graph_state": state["graph_state"] + " a student"}

def node3(state: State) -> State:
    print ("--- Node 3 executed ---")
    return {"graph_state": state["graph_state"] + " a teacher"}

# Define a decision node that will route the flow to either node2 or node3 based on a random condition. In a real scenario, this could be based on user input or any other condition in the state.
def decide_node_path(state: State) -> Literal["nodo2", "nodo3"]:
    print ("--- Decision Node executed ---")
    user_input = state["graph_state"]

    if random.random() < 0.5:
        return "nodo2"
    return "nodo3"

# Save the graph as a PNG file
def save_graph(path: str, graph: bytes):
    file_exist = Path(path).exists()
    if not file_exist:
        png_graph = graph.get_graph().draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png_graph)
    else:
        print(f"Graph already exists at {path}, skipping save.")

# Graph definition
builder = StateGraph(State)
builder.add_node("nodo1", node1)
builder.add_node("nodo2", node2)
builder.add_node("nodo3", node3)

# Logical connections
builder.add_edge(START, "nodo1")
builder.add_conditional_edges("nodo1", decide_node_path)
builder.add_edge("nodo2", END)
builder.add_edge("nodo3", END)

# Build the graph
graph = builder.compile()

# Save and open the graph
save_graph("simple_graph.png", graph)

# Execute the graph with an initial state
state = graph.invoke({"graph_state": "Hello, this is Ed. "})
print (state["graph_state"])
