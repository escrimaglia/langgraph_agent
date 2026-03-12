# Simple agent that uses the Chain pattern
# By Ed SCrimaglia

from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AnyMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from pathlib import Path

class State(TypedDict):
    messages : Annotated[List[AnyMessage], add_messages]

load_dotenv()

# LLM Initialization
llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    temperature=0.2,
    max_completion_tokens=2000,
    max_tokens=5000,
    api_version="2024-10-21"
)

# Nodo
def calling_llm(state: State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}

def save_graph(path: str, graph: bytes):
    file_file = Path(path).exists()
    if not file_file:
        png_graph = graph.get_graph().draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png_graph)
    else:
        print(f"Graph already exists at {path}, skipping save.")

if __name__ == "__main__":
    # Graph definition
    builder = StateGraph(State)
    builder.add_node("calling_llm", calling_llm)

    # Logical connections
    builder.add_edge(START, "calling_llm")
    builder.add_edge("calling_llm", END)
    graph = builder.compile()

    # Save the Graph
    save_graph("chain_graph.png", graph)

    state = graph.invoke({"messages": [HumanMessage(content="Hola, ¿cómo estás?")]})
    state = graph.invoke({"messages": state["messages"] + [HumanMessage(content="me puedes contar sobre la region de los Sudetes en la WWII?")]})

    # All the responses are stored in the state
    print ("\n--- All messages accumulated in the state ---")
    print (f"Total messages: {len(state['messages'])}")
    for i, msg in enumerate(state["messages"]):
        print(f"{i}: {type(msg).__name__}: {msg.content}")