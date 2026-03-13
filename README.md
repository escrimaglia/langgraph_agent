# LangGraph and LangChain Agents

This repository is a hands-on learning collection for beginners who want to understand how to build agent workflows with **LangGraph** and **LangChain**.

The examples move from basic graph execution to tool-calling agents, typed state schemas, memory strategies, and parallel execution patterns.

## Scope of this README

Included folders:

- `simple_chain/`
- `simple_graph/`
- `simple_agent/`
- `simple_router/`
- `schema/`
- `memory/`
- `memory_summary/`
- `paralleliization/`

---

## Prerequisites

- Python `3.12+`
- Environment variables for Azure OpenAI in `.env`
- Dependencies from `pyproject.toml`

Install dependencies with your preferred tool (for example `uv` or `pip`) and run scripts from the repository root.

---

## Learning Path

If you are new to agent programming, this order is recommended:

1. `simple_chain/simple_chain.py`
2. `simple_graph/simple_graph.py`
3. `simple_router/router.py`
4. `simple_agent/simple_agent.py`
5. `schema/state_schema_no_tools.py`
6. `schema/state_multi_schema.py`
7. `schema/state_schema_tools.py`
8. `memory/internal_memory.py`
9. `memory/external_memory.py`
10. `memory_summary/memory_summary.py`
11. `paralleliization/no_parallel.py`
12. `paralleliization/parallel.py`
13. `paralleliization/custom_parallel.py`

---

## Folder-by-Folder Script Analysis

## `simple_chain/`

### `simple_chain/simple_chain.py`

What it teaches:

- The most basic LangGraph agent flow: `START -> LLM node -> END`
- How to store chat history in state using `add_messages`
- How to invoke a graph multiple times while keeping context

How it works:

- Defines a `State` typed dictionary with a `messages` field.
- Creates one node (`calling_llm`) that calls `AzureChatOpenAI`.
- Compiles and executes the graph.
- Sends two user prompts so you can see accumulated message history.
- Exports a graph image (`chain_graph.png`).

Why it matters for beginners:

- It is the shortest path to understanding graph-based orchestration.

---

## `simple_graph/`

### `simple_graph/simple_graph.py`

What it teaches:

- Control flow with conditional routing in LangGraph
- Non-LLM graph nodes for deterministic state transformations

How it works:

- Uses `graph_state` text in state.
- Runs `node1`, then a decision function selects `node2` or `node3`.
- Ends after one of the two branches.
- Exports `simple_graph.png`.

Why it matters for beginners:

- Shows that LangGraph is not only for LLM calls, but for workflow logic too.

---

## `simple_agent/`

### `simple_agent/simple_agent.py`

What it teaches:

- Building a tool-calling agent with a loop (`LLM -> tools -> LLM`)
- Tool binding with `llm.bind_tools(...)`
- Automatic routing with `tools_condition`

How it works:

- Defines three math tools: multiply, sum, divide.
- Uses `ToolNode` to execute tool calls requested by the model.
- Routes based on whether the model asks for a tool.
- Ends when no further tool call is needed.
- Exports `agent_graph.png`.

Why it matters for beginners:

- This is the core architecture of many practical agent systems.

---

## `simple_router/`

### `simple_router/router.py`

What it teaches:
- Router pattern: one input can go through tool execution or direct response

How it works:

- Binds one tool (`multiplay`) to the model.
- If input requires a tool, flow goes to `ToolNode`.
- Otherwise, it ends after the LLM answer.
- Demonstrates both routes with two prompts.
- Exports `router.png`.

Why it matters for beginners:

- Helps you reason about branching behavior in mixed queries.

---

## `schema/`

### `schema/state_schema_no_tools.py`

What it teaches:

- State validation with Pydantic
- Structured state fields beyond just message lists

How it works:

- Defines `PydanticState` with constrained literals (`name`, `humor`) and `messages`.
- `parse_input` creates a system message based on structured fields.
- `calling_llm` uses that context to respond.
- Exports `schema_no_tools.png`.

Why it matters for beginners:

- Introduces typed, validated state design for more robust agents.

### `schema/state_multi_schema.py`

What it teaches:

- Separate input, internal, and output schemas
- Encapsulation of private state fields

How it works:

- Uses three models:
  - `PrivateState` (input)
  - `AllStates` (internal graph state)
  - `OutputState` (final output)
- Parses private data first, then calls the LLM.
- Exports `multi_schema.png`.

Why it matters for beginners:

- Teaches clean contracts between user input and internal graph logic.

### `schema/state_schema_tools.py`

What it teaches:

- Tool calling with schema-driven state
- Manual tool execution from `tool_calls`

How it works:

- Defines tools (`get_person_name`, `get_person_humor`) and binds them to the model.
- Reads requested tool calls from the last model message.
- Executes tools against current state fields.
- Returns `ToolMessage` objects and loops back to the model.
- Exports `schema_tool.png`.

Why it matters for beginners:

- Shows the internals of tool handling, not only prebuilt abstractions.

---

## `memory/`

### `memory/internal_memory.py`

What it teaches:

- In-memory checkpointing with `MemorySaver`
- Message trimming with `RemoveMessage` to control token growth

How it works:

- Builds a math tool agent with in-memory persistence.
- Reuses a thread ID so consecutive calls share history.
- Adds a filtering node that drops old messages while preserving recent context.
- Serializes memory to a JSON file for inspection.
- Exports `memory_graph.png`.

Why it matters for beginners:

- Demonstrates short-term conversational memory and context window management.

### `memory/external_memory.py`

What it teaches:

- Persistent memory in SQLite with `SqliteSaver`

How it works:

- Uses a SQLite DB (`default_memory.db`) as checkpointer backend.
- Keeps conversation state across multiple invocations with the same thread ID.
- Runs a sequence of prompts to demonstrate persisted identity and history.
- Exports `memory_graph.png`.

Why it matters for beginners:

- Essential pattern when your agent must survive process restarts.

---

## `memory_summary/`

### `memory_summary/memory_summary.py`

What it teaches:

- Memory persistence + summarization strategy via pruning/filtering
- Interactive command-line chat loop with a LangGraph agent

How it works:

- Uses `MessagesState` instead of custom `TypedDict`.
- Persists checkpoints in SQLite.
- Routes to tools when needed; otherwise routes to a filtering node.
- Filtering node removes old messages while keeping recent context.
- Runs in a `while True` prompt loop until `exit`/`quit`.
- Exports `memory_summary.png`.

Why it matters for beginners:

- Introduces practical memory hygiene needed for long-running assistants.

---

## `paralleliization/`

### `paralleliization/no_parallel.py`

What it teaches:
- Baseline sequential graph execution

How it works:

- Executes nodes `Node1 -> Node2 -> Node3 -> Node4 -> Node5` in order.
- Each node updates state with its own value.
- Exports `no_parallel.png`.

Why it matters for beginners:

- Gives a baseline to compare against fan-out/fan-in patterns.

### `paralleliization/parallel.py`

What it teaches:

- Native parallel branches in LangGraph
- Reducer usage (`operator.add`) for combining concurrent updates

How it works:

- After `Node1`, graph fans out to `Node2` and `Node3` in parallel.
- Both converge into `Node4`.
- State field is annotated with reducer to merge outputs safely.
- Exports `parallel.png`.

Why it matters for beginners:

- Demonstrates concurrency and state merge fundamentals.

### `paralleliization/custom_parallel.py`

What it teaches:

- Custom reducers for deterministic merge logic in parallel flows

How it works:

- Defines `custom_reducer(left, right)` that combines and sorts values.
- Uses a slightly deeper branch (`Node2 -> Node2b`) before merge.
- Converges with `Node3` into `Node4`.
- Exports `custom_parallel.png`.

Why it matters for beginners:

- Shows how to design conflict resolution policies for concurrent state updates.

---

## Common Patterns You Will See Across Scripts

- `StateGraph(...)` to define the workflow
- `START` and `END` as explicit graph boundaries
- Message reducers (`add_messages`) to accumulate conversation history
- `ToolNode` + `tools_condition` for tool-calling loops
- Checkpointers (`MemorySaver`, `SqliteSaver`) for memory persistence
- Graph visualization through `draw_mermaid_png()`

---

## Suggested Next Steps for Learners

1. Replace math tools with domain tools (for example: CRM, tickets, docs search).
2. Add error handling and retries around tool execution.
3. Add evaluation scripts to compare answer quality with and without memory pruning.
4. Convert one script into a production-ready service (API + persistent storage + logs).

### `Ed Scrimaglia`
