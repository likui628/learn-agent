# Learn Agent
 
A demonstration project showcasing the core patterns of AI coding agents, ported from [learn.shareai.run](https://learn.shareai.run/en/s01/). The original tutorial uses the **Anthropic SDK** directly; this project migrates to the **OpenAI-compatible interface via OpenRouter**, enabling model-agnostic usage.
 
## Installation
 
Ensure you have Python 3.12+ installed. This project uses `uv` for dependency management.
 
```bash
git clone https://github.com/likui628/learn-agent
cd learn-agent
uv sync
```
 
## Environment Setup
 
Create a `.env` file in the root directory:
 
```
OPENROUTER_API_KEY=your_openrouter_api_key
MODEL_ID=anthropic/claude-haiku-4-5   # optional, defaults to claude-haiku-4-5
```

## How to Run
 
```bash
uv run -m agents.s01_agent_loop
```
 
Try these prompts to get started:
 
```
s01 >> Create a file called hello.py that prints "Hello, World!"
s01 >> List all Python files in this directory
s01 >> What is the current git branch?
s01 >> Create a directory called test_output and write 3 files in it
```
 
## Anthropic SDK vs OpenAI-Compatible Interface
 
The original tutorial calls the Anthropic API directly. This project routes through OpenRouter using the OpenAI-compatible SDK. Below are the key differences across each part of the agent loop.
 
### 1. Client Initialization
 
| | Anthropic SDK (original) | OpenAI-compatible (this project) |
|---|---|---|
| **Import** | `import anthropic` | `from openai import OpenAI` |
| **Base URL** | (default Anthropic endpoint) | `https://openrouter.ai/api/v1` |
| **API Key env var** | `ANTHROPIC_API_KEY` | `OPENROUTER_API_KEY` |
 
```python
# Anthropic SDK
import anthropic
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
 
# OpenAI-compatible (this project)
from openai import OpenAI
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
```
 
### 2. Tool Definition
 
Anthropic uses a top-level `input_schema` field. The OpenAI format wraps the tool inside a `type: "function"` object and names the schema field `parameters`.
 
```python
# Anthropic SDK
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }
}]
 
# OpenAI-compatible (this project)
TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]
```
 
### 3. Sending Messages & Checking Stop Reason
 
Two important differences here:
 
- **System prompt**: In the Anthropic SDK it is a top-level `system` parameter on `client.messages.create()`; in the OpenAI format it is prepended to the `messages` list as a `role: "system"` entry.
- **Stop signal**: The field name (`stop_reason` vs `finish_reason`) and its value (`"tool_use"` vs `"tool_calls"`) differ between the two APIs.
```python
# Anthropic SDK
response = client.messages.create(
    model=MODEL, system=SYSTEM,
    messages=messages, tools=TOOLS, max_tokens=8000,
)
if response.stop_reason != "tool_use":
    return  # model is done
 
# OpenAI-compatible (this project)
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "system", "content": SYSTEM}] + messages,
    tools=TOOLS, max_tokens=8000,
)
msg = response.choices[0].message
if response.choices[0].finish_reason != "tool_calls":
    return  # model is done
```
 
### 4. Parsing Tool Calls & Writing Results Back
 
This is the most critical difference — the "write-back" step that closes the loop.
 
Anthropic returns tool calls as typed content blocks inside `response.content`, and results are written back as a `role: "user"` message containing `tool_result` blocks.
 
The OpenAI format exposes tool calls via `message.tool_calls`, and each result is written back as an independent `role: "tool"` message keyed by `tool_call_id`.
 
```python
# Anthropic SDK — tool results go back as a user message with content blocks
for block in response.content:
    if block.type == "tool_use":
        output = run_bash(block.input["command"])
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        })
messages.append({"role": "user", "content": results})
 
# OpenAI-compatible (this project) — each result is a standalone "tool" message
for tc in msg.tool_calls:
    command = json.loads(tc.function.arguments)["command"]
    output = run_bash(command)
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": output,
    })
```
 
### Summary
 
| Aspect | Anthropic SDK | OpenAI-compatible |
|---|---|---|
| System prompt | Top-level `system=` param | `{"role": "system", ...}` in messages |
| Tool schema field | `input_schema` | `parameters` (inside `function`) |
| Tool wrapper | None (flat object) | `{"type": "function", "function": {...}}` |
| Stop signal field | `stop_reason` | `finish_reason` |
| Stop signal value | `"tool_use"` | `"tool_calls"` |
| Tool call location | `response.content` blocks | `response.choices[0].message.tool_calls` |
| Tool result role | `"user"` (with content blocks) | `"tool"` (standalone message) |
| Result ID field | `tool_use_id` | `tool_call_id` |
 
## Architecture
 
```
+----------+      +-------+      +---------+
|   User   | ---> |  LLM  | ---> |  Tool   |
|  prompt  |      |       |      | execute |
+----------+      +---+---+      +----+----+
                      ^               |
                      |   tool_result |
                      +---------------+
                 (loop until finish_reason != "tool_calls")
```
 
The agent loop keeps running until the model stops requesting tools. Every tool result is appended back into the message history so the model always has full context of prior actions.
 
## Project Structure
 
```
learn-agent/
├── agents/
│   └── s01_agent_loop.py   # The core agent loop
├── .env                     # API keys (not committed)
├── pyproject.toml
└── README.md
```