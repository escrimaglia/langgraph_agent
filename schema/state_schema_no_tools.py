# This example demonstrates how to use a state graph without tools. 
# I define a state schema that includes messages exchanged in the conversation. 
# Pydantic is used to define the state schema, and the graph processes the input state, generates messages, and then invokes an LLM to produce a response based on those messages.
# The graph processes the input state, generates messages, and then invokes an LLM to produce a response based on those messages.
# By Ed Scrimaglia

from typing import List, Annotated, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langgraph.graph import START, END, StateGraph
from langchain_openai import AzureChatOpenAI
from langgraph.graph.message import add_messages
from pathlib import Path

load_dotenv()

# Define the state with multiple schemas
class PydanticState(BaseModel):
    name: Literal["Ed", "Juan"] = Field(..., description="The name of the state")
    humor: Literal["feliz", "triste"] = Field(..., description="The humor of the state")
    messages: Annotated[List[AnyMessage], add_messages] = Field(..., description="Messages exchanged in the conversation")


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

# Node to invoke the LLM
def calling_llm(state: PydanticState) -> dict:
    response = llm.invoke(state.messages)
    return {"messages": [response]}

# Node to parse Input Message
def parse_input(state: PydanticState) -> PydanticState:
    """
    Nodo que analiza y procesa los campos del estado.
    """
    parsed_result = f"{state.name} está {state.humor}"
    # Agregar el resultado como un mensaje del sistema
    info_message = SystemMessage(content=f"Estado procesado: {parsed_result}")
    return {"messages": [info_message]}

# Build the Graph
builder = StateGraph(PydanticState)
builder.add_node("parse_input", parse_input)
builder.add_node("calling_llm", calling_llm)

# Define edges - primero procesa, luego el LLM responde
builder.add_edge(START, "parse_input")
builder.add_edge("parse_input", "calling_llm")
builder.add_edge("calling_llm", END)
graph = builder.compile()

# Save the Graph
save_graph("schema_no_tools.png", graph)

# Use the graph with initial messages
try:
    state = graph.invoke(PydanticState(name="Ed", humor="feliz", messages=[HumanMessage(content="Hola, puedes decirme el nombre y el humor de la persona?")]))
    for message in state["messages"]:
        message.pretty_print()
except Exception as error:
    print(f"Error invoking graph: {error}")