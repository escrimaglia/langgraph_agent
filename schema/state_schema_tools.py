# This example demonstrates how to use tools within a state graph. 
# I define a state schema that includes messages exchanged in the conversation. 
# The graph processes the input state, generates messages, and then invokes an LLM to produce a response based on those messages.
# Pydantic is used to define the state schema, and the graph includes nodes that can call tools to access specific information from the state.
# The LLM can also call tools to access specific information from the state, and the graph will route the messages accordingly.
# By Ed Scrimaglia

from typing import List, Dict, Annotated, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from langchain_openai import AzureChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
import json
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

# Tools without parameters to access the state
def get_person_name():
    """
    Obtiene el nombre de la persona.
    
    Returns:
        str: El nombre de la persona.
    """
    return "name"

def get_person_humor():
    """
    Obtiene el humor de la persona.
    
    Returns:
        str: El humor de la persona.
    """
    return "humor"

system_msg = SystemMessage(content="Eres un asistente que ayuda a obtener información sobre una persona. Usa las herramientas get_person_name y get_person_humor para obtener el nombre y el humor, y luego responde al usuario.")

# LLM Initialization
llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    temperature=0.2,
    max_completion_tokens=2000,
    max_tokens=5000,
    api_version="2024-10-21"
)

tools = [get_person_name, get_person_humor]
llm_with_tools = llm.bind_tools(tools)

# Node to call the LLM with tools
def calling_llm(state: PydanticState) -> dict:
    return {"messages": [llm_with_tools.invoke([system_msg] + state.messages)]}

# Node to execute tools based on the state
def execute_tools(state: PydanticState) -> dict:
    """
    Ejecuta los tools solicitados y devuelve los resultados basados en el estado actual.
    """
    messages = state.messages
    last_message = messages[-1]
    
    tool_messages = []
    
    # Process the tool calls in the last message
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_call_id = tool_call["id"]
        
        # Execute the appropriate tool based on the current state
        if tool_name == "get_person_name":
            result = state.name
        elif tool_name == "get_person_humor":
            result = state.humor
        else:
            result = f"Tool {tool_name} no encontrado"
        
        # Create a ToolMessage with the result and the tool call ID
        tool_message = ToolMessage(
            content=json.dumps({"result": result}),
            tool_call_id=tool_call_id,
            name=tool_name
        )
        tool_messages.append(tool_message)
    
    return {"messages": tool_messages}

# Build the Graph
builder = StateGraph(PydanticState)
builder.add_node("calling_llm", calling_llm)
builder.add_node("tools", execute_tools)

# Define edges 
builder.add_edge(START, "calling_llm")
builder.add_conditional_edges(
    "calling_llm",
    tools_condition,
)
builder.add_edge("tools", "calling_llm")
builder.add_edge("calling_llm", END)
graph = builder.compile()

# Save the Graph
save_graph("schema_tool.png", graph)

# Use the graph with initial messages
try:
    state = graph.invoke(PydanticState(name="Ed", humor="sad", messages=[HumanMessage(content="Hola, puedes decirme el nombre y el humor de la persona?")]))
    for message in state["messages"]:
        message.pretty_print()
except Exception as error:
    print(f"Error invoking graph: {error}")